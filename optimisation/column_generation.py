from multiprocessing.sharedctypes import Value
import numpy as np
from math import log, exp
from scipy import stats
from time import time
from service_class.service import Service
from service_class.graph import service_graph, service_path
from topology.network import Network
from topology.location import Dummy, Node
import gurobipy as gp
from gurobipy import GRB
from scipy import optimize
import logging

inf = 1000000000
eps = 1e-9
env = gp.Env(empty=True)
env.setParam("OutputFlag",0)
env.start()

class ColumnGeneration(object):
    """
    Description:    Class representing the datacenter optimisation problem via column generation.
    Parameters:     datacenter - Instance of Datacenter class to be optimised.
                    services - list of Service class representing the services to be hosted in the datacenter.
                    verbose - int verbosity of output.
                    results - string filename to use to log output.
    """
    def __init__(self, network: Network, services: list, verbose: int = 1, logfile: str ='log.txt', max_replicas: int = 5,
                node_availability: float = 0.99, min_flow_param: int = 10) -> None:
        """
        Initialises the optimisation problem
        """
        if verbose not in [0, 1, 2]:
            raise ValueError("Invalid verbosity level. Use 0 for no log, 1 for info and 2 for info and debug.")
        self.verbose = verbose
        logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.DEBUG, filemode='w') if self.verbose == True else None

        self.network = network

        # Adds the dummy node to the network.
        self.network.add_dummy_node()

        self.services = services
        self.lower_bound = eps
        self.upper_bound = inf
        self.status = None
        self.model = None
        self.max_replicas = 5
        self.node_availability = 0.99
        self.min_flow_param = 10
        
        # Initialises inner approximation for each service
        for service in services:
            service.make_graph(self.network)

    def pricing_problem(self, service):
        """
        Solves the pricing problem for a service and finds the best new path to add to the master problem.
        """
        graph = service.graph
        m = gp.Model(graph.description, env=env)
        w = np.array([l.cost for l in graph.links])
        x = m.addMVar(shape = len(graph.links), name = [l.get_description() for l in graph.links], vtype = GRB.BINARY)

        # Gets source and sink
        source = graph.get_location_by_description(service.source.description + "_l0")
        sink = graph.get_location_by_description(service.sink.description + "_l{}".format(graph.n_layers - 1))

        # Flow constraints for each node.
        for node in graph.locations:
            o_indexes = [graph.links.index(o) for o in graph.outgoing_edge(node)]
            i_indexes = [graph.links.index(i) for i in graph.incoming_edge(node)]
            # 1 edge leaving the source must be active.
            if node == source:
                m.addConstr(gp.quicksum(x[o] for o in o_indexes) - gp.quicksum(x[i] for i in i_indexes) == 1, name = "sourceflow_{}".format(node.description))
            # 1 edge entering sink must be active
            elif node == sink:
                m.addConstr(gp.quicksum(x[o] for o in o_indexes) - gp.quicksum(x[i] for i in i_indexes) == -1, name = "sinkflow_{}".format(node.description))
            # Flow conservation for every other node.
            else:
                m.addConstr(gp.quicksum(x[o] for o in o_indexes) == gp.quicksum(x[i] for i in i_indexes), name = "conservation_{}".format(node.description))
        
        # Adds latency constraint.
        lat = np.array([l.latency for l in graph.links])
        m.addConstr(lat @ x <= service.latency, name = "latency")

        # Gets dual variable associated with constraint that 100% of the flow must be used
        if self.model != None:
            pi = -self.model.getConstrByName("pathflow_{}".format(service.description)).getAttr("Pi")
        else:
            pi = 0

        # Sets objective to reduced cost.
        m.setObjective(pi + w @ x, GRB.MINIMIZE)
        m.update()
        m.optimize()
        m.write("{}.lp".format(m.getAttr("ModelName")))
        if m.status == GRB.OPTIMAL:
            logging.info(" Optimisation terminated successfully.") if self.verbose > 0 else None
            logging.info(' Ojective: {}'.format(m.objVal)) if self.verbose > 0 else None
            logging.info(' Vars:')
            vars = m.getVars()
            for i in range(len(vars)):
                if vars[i].x != 0:
                    logging.info("Description: {}, Value: {}".format(graph.links[i].get_description(), vars[i].x)) if self.verbose > 0 else None
            return m
        else:
            logging.error(" Optimisation Failed - consult .ilp") if self.verbose > 0 else None
            m.computeIIS()
            m.write("{}.ilp".format(m.getAttr("ModelName")))
            raise ValueError("Optimisation failed")

    def add_path_from_model(self, service, model):
        """
        Given a service and a column generation model it adds the current path to the service graph.
        """
        graph = service.graph
        vars = model.getVars()
        # Gets the list of edges used - equal to the edges whose binary variable has value 1.
        used_edges = [graph.links[i] for i in range(len(graph.links)) if vars[i].x == 1]

        # Gets the list of used nodes from the used edges.
        used_nodes = []
        for edge in used_edges:
            if edge.source not in used_nodes: used_nodes.append(edge.source)
            if edge.sink not in used_nodes: used_nodes.append(edge.sink)

        # Makes the path and adds it to the list of paths for the service graph.
        path = service_path(service.description, used_nodes, used_edges, self.network, service, n_layers = graph.n_layers)
        graph.add_path(path)
        return path

    def build_initial_model(self):
        """
        Builds and solves the restricted master problem given the initial approximation points.
        """
        self.model = gp.Model(self.network.description, env=env)
        nodes = [n for n in self.network.locations if isinstance(n, Node) == True]
        # Finds the dummy node to ensure that availability is satisfied if dummy node is used.
        dummy_node = [n for n in nodes if isinstance(n, Dummy)]
        # Checks that there is only one dummy node.
        assert len(dummy_node) <= 1
        dummy_node = dummy_node[0]

        vnfs = set()
        # Also makes set of components for all services so that no duplicate components are considered
        for service in self.services:
            for f in service.vnfs:
                vnfs.add(f)
        vnfs = list(vnfs)

        # Gets set of edges - not including reverse edges since edges are bidirectional.
        edges = []
        for edge in self.network.links:
            if edge.get_description() and edge.get_opposing_edge_description() not in [e.get_description() for e in edges]:
                edges.append(edge)

        # Adds variables, for LP relaxation makes variables continuous.
        # Assignment variables.
        for node in nodes:
            for function in vnfs:
                self.model.addVar(lb = 0, ub = 1, vtype=GRB.CONTINUOUS, obj = node.cost, name= node.description + "_" + function.description + "_assignment")
        
        # Path variables.
        for service in self.services:
            for path in service.graph.paths:
                self.model.addVar(vtype=GRB.CONTINUOUS, name = path.description + "_flow")
        
        # Service installation variables
        for service in self.services:
            for node in nodes:
                for function in service.vnfs:
                    self.model.addVar(lb = 0, ub = 1, vtype=GRB.CONTINUOUS, name= service.description + "_" + node.description + "_" + function.description + "_installation")

        # Service replication variables
        for service in self.services:
            for function in service.vnfs:
                for i in range(1, self.max_replicas + 1):
                    self.model.addVar(lb = 0, ub = 1, vtype=GRB.CONTINUOUS, name= service.description + "_" + function.description + "_" + str(i) + "_replication")

        self.model.update()

        # Adds constraint that the sum of CPU and RAM of the functions does not exceed the capacity of the nodes.
        for node in nodes:
            cpu_n, ram_n = node.cpu, node.ram
            cpu_f, ram_f = [f.cpu for f in vnfs], [f.ram for f in vnfs]
            vars_used = [self.model.getVarByName(node.description + "_" + f.description + "_assignment") for f in vnfs]
            self.model.addConstr(gp.quicksum(cpu_f[i] * vars_used[i] for i in range(len(vnfs))) <= cpu_n, name = "cpu_{}".format(node.description))
            self.model.addConstr(gp.quicksum(ram_f[i] * vars_used[i] for i in range(len(vnfs))) <= ram_n, name = "ram_{}".format(node.description))

        # Adds constraint that says the sum of all flows through each edge must not exceed the bandwidth capacity.
        for edge in edges:
            e_ij, e_ji = edge.get_description(), edge.get_opposing_edge_description()
            links_used, tp, vars_used = [], [], []
            for s in self.services:
                for p in s.graph.paths:
                    links_used.append(p.get_params()["times traversed"][e_ij] + p.get_params()["times traversed"][e_ji])
                    tp.append(s.throughput)
                    vars_used.append(self.model.getVarByName(p.description + "_flow"))
            self.model.addConstr(gp.quicksum(links_used[i] * tp[i] * vars_used[i] for i in range(len(vars_used))) <= edge.bandwidth, name = "bandwidth_{}".format(edge.get_description()))

        # Adds a constraint that says for each service, 100% of the flow must be routed.
        for service in self.services:
            vars_used = [self.model.getVarByName(p.description + "_flow") for p in service.graph.paths]
            self.model.addConstr(gp.quicksum(vars_used) == 1, name = "pathflow_{}".format(service.description))

        # Adds a constraint that says, if a vnf is not installed on a node, then every path that considers the vnf to be installed on that node must have zero flow.
        for service in self.services:
            for node in nodes:
                for function in service.vnfs:
                    assignment_var_used = self.model.getVarByName(node.description + "_" + function.description + "_assignment")
                    flow_vars_used = [self.model.getVarByName(p.description + "_flow") for p in service.graph.paths]
                    assignment_params = [1 if p.get_params()["components assigned"][function.description] == node.description else 0 for p in service.graph.paths]
                    self.model.addConstr(gp.quicksum(assignment_params[i] * flow_vars_used[i] for i in range(len(service.graph.paths))) <= assignment_var_used, name="assignment_{}_{}_{}".format(service.description, node.description, function.description))
        
        # Adds a constraint that forces the installation variable of a service to be 1 if at least one path using that node is used to route flow.
        for service in self.services:
            for node in nodes:
                for function in service.vnfs:
                    installation_var_used = self.model.getVarByName(service.description + "_" + node.description + "_" + function.description + "_installation")
                    flow_vars_used = [self.model.getVarByName(p.description + "_flow") for p in service.graph.paths]
                    assignment_params = [1 if p.get_params()["components assigned"][function.description] == node.description else 0 for p in service.graph.paths]
                    self.model.addConstr(self.min_flow_param * gp.quicksum(assignment_params[i] * flow_vars_used[i] for i in range(len(service.graph.paths))) >= installation_var_used, name="installation_{}_{}_{}".format(service.description, node.description, function.description))
        
        # Adds a constraint that forces one of the replication variables to take a value of one.
        for service in self.services:
            for function in service.vnfs:
                dummy_installation = self.model.getVarByName(service.description + "_" + dummy_node.description + "_" + function.description + "_installation")
                vars_used = [self.model.getVarByName(service.description+ "_" + function.description + "_" + str(i) + "_replication") for i in range(1, self.max_replicas + 1)]
                self.model.addConstr(gp.quicksum(vars_used) + dummy_installation == 1, name = "replication_{}_{}".format(service.description, function.description))
        
        # Adds a constraint that constrains the number of replicas to be equal to the number of different nodes hosting that function across the service paths.
        for service in self.services:
            for function in service.vnfs:
                #dummy_installation = self.model.getVarByName(service.description + "_" + dummy_node.description + "_" + function.description + "_installation")
                replication_vars_used = [self.model.getVarByName(service.description+ "_" + function.description + "_" + str(i) + "_replication") for i in range(1, self.max_replicas + 1)]
                installation_var_used = [self.model.getVarByName(service.description + "_" + n.description + "_" + function.description + "_installation") for n in nodes if isinstance(n, Dummy) == False]
                params = [i for i in range(1, self.max_replicas + 1)]
                self.model.addConstr(gp.quicksum(params[i] * replication_vars_used[i] for i in range(len(replication_vars_used))) <= gp.quicksum(installation_var_used), name = "nreplicas_{}_{}".format(service.description, function.description))
        
        # Adds availability constraints for each service.
        for service in self.services:
            rhs = log(service.availability)
            vars_used, params, dummy_installations = [], [], []
            for function in service.vnfs:
                for i in range(1, self.max_replicas + 1):
                    vars_used.append(self.model.getVarByName(service.description+ "_" + function.description + "_" + str(i) + "_replication"))
                    params.append(-(1 - self.node_availability * function.availability)**i)
                dummy_installations.append(self.model.getVarByName(service.description + "_" + dummy_node.description + "_" + function.description + "_installation"))
            self.model.addConstr(gp.quicksum(params[i] * vars_used[i] for i in range(len(vars_used))) + gp.quicksum(dummy_installations[i] for i in range(len(service.vnfs)))>= rhs, name = "availability_{}".format(service.description))

        # Updates and optimises the model.
        self.model.update()
        self.model.optimize()
            
        logging.info(" Initial model built, solving.") if self.verbose > 0 else None
        self.model.update()
        self.model.write("{}.lp".format(self.model.getAttr("ModelName")))
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            self.upper_bound = self.model.objVal
            logging.info( " Optimisation terminated successfully.") if self.verbose > 0 else None
            logging.info(' Objective: {}'.format(self.model.objVal)) if self.verbose > 0 else None
            logging.info(' Vars:') if self.verbose > 0 else None
            for v in self.model.getVars():
                if v.x != 0:
                    logging.info(" Variable {}: ".format(v.varName) + str(v.x)) if self.verbose > 0 else None
            logging.info(" Dual Solution:") if self.verbose > 0 else None
            for c in self.model.getConstrs():
                logging.info(" Constr {}: ".format(c.getAttr("ConstrName")) + str(c.getAttr("Pi"))) if self.verbose > 0 else None
        else:
            logging.error(" Optimisation Failed - consult .ilp file") if self.verbose > 0 else None
            self.model.computeIIS()
            self.model.write("{}.ilp".format(self.model.getAttr("ModelName")))
            raise ValueError("Optimisation failed")
    
    def rmp(self):
        """
        Updates and solves the RMP and prints primal and dual solution to log.
        """
        self.model.update()
        self.model.optimize()
        self.model.write("{}.lp".format(self.model.getAttr("ModelName")))
        if self.model.status == GRB.OPTIMAL:
            self.upper_bound = self.model.objVal
            logging.info( " Optimisation terminated successfully.") if self.verbose > 0 else None
            logging.info(' Objective: {}'.format(self.model.objVal)) if self.verbose > 0 else None
            logging.info(' Vars:') if self.verbose > 0 else None
            for v in self.model.getVars():
                if v.x != 0:
                    logging.info(" Variable {}: ".format(v.varName) + str(v.x)) if self.verbose > 0 else None
            logging.info(" Dual Solution:") if self.verbose > 0 else None
            for c in self.model.getConstrs():
                logging.info(" Constr {}: ".format(c.getAttr("ConstrName")) + str(c.getAttr("Pi"))) if self.verbose > 0 else None
        else:
            logging.error(" Optimisation Failed - consult .ilp file") if self.verbose > 0 else None
            self.model.computeIIS()
            self.model.write("{}.ilp".format(self.model.getAttr("ModelName")))
            raise ValueError("Optimisation failed")

    def update_duals(self):
        """
        Updates the edge weights of the service graphs given the dual values from the current iteration.
        """
        for service in self.services:
            graph = service.graph
            normal_edges = [e for e in graph.links if e.assignment_link == False]
            # Updates the edge cost using the dual value associated with the bandwidth constraint for each link.
            for e in normal_edges:
                e_o = graph.get_edge_from_original_network(e)
                if self.model.getConstrByName("bandwidth_{}".format(e_o.get_description())) != None:
                    e.cost = -self.model.getConstrByName("bandwidth_{}".format(e_o.get_description())).getAttr("Pi") * service.throughput
                elif self.model.getConstrByName("bandwidth_{}".format(e_o.get_opposing_edge_description())) != None:
                    e.cost = -self.model.getConstrByName("bandwidth_{}".format(e_o.get_opposing_edge_description())).getAttr("Pi") * service.throughput
                else:
                    raise KeyError("Bandwidth constraint not found for edge.")
            # Updates the assignment edge cost using the duals associated with the assignment constraints.
            assignment_edges = [e for e in graph.links if e.assignment_link == True]
            for e in assignment_edges:
                node, function = graph.get_node_and_function_from_assignment_edge(e)
                constr1 = self.model.getConstrByName("assignment_{}_{}_{}".format(service.description, node.description, function.description))
                constr2 = self.model.getConstrByName("installation_{}_{}_{}".format(service.description, node.description, function.description))
                if constr1 != None and constr2 != None:
                    e.cost = -(constr1.getAttr("Pi") + self.min_flow_param * constr2.getAttr("Pi"))
                else:
                    raise KeyError("Constraint not found for this service, node, function combination.")
    
    def add_column_from_path(self, path):
        """
        Adds a new path variable with column of coefficients taken from the pricing problem.
        """
        params = path.get_params()
        constrs = self.model.getConstrs()
        # Gets the constraints containing the variable.
        bw = [c for c in constrs if "bandwidth" in c.getAttr("ConstrName")]
        pf = self.model.getConstrByName("pathflow_{}".format(path.service.description))
        ass = [c for c in constrs if "assignment_{}".format(path.service.description) in c.getAttr("ConstrName")]
        inst = [c for c in constrs if "installation_{}".format(path.service.description) in c.getAttr("ConstrName")]
        constrs = bw + [pf] + ass + inst
        coefficients = np.zeros(len(constrs))

        # Gets the coefficients in the column associated with the new path variable in each constraint.
        for i in range(len(constrs)):
            if "bandwidth" in constrs[i].getAttr("ConstrName"):
                # For bandwidth constraints this is the number of times the link is used multiplied by the throughput of the service.
                edge = self.network.get_link_by_description(constrs[i].getAttr("ConstrName").split("_")[-1])
                opp_edge = edge.get_opposing_edge_description()
                edge = edge.get_description()
                coefficients[i] = (params["times traversed"][edge] + params["times traversed"][opp_edge]) * path.service.throughput
            elif "pathflow" in constrs[i].getAttr("ConstrName"):
                # This is just 1 since we sum all the path flow variables.
                coefficients[i] = 1
            elif "assignment" in constrs[i].getAttr("ConstrName"):
                # This is 1 if the path has the function hosted on a particular node.
                tokens = constrs[i].getAttr("ConstrName").split("_")
                node, function = tokens[2], tokens[3]
                if params["components assigned"][function] == node:
                    coefficients[i] = 1
                else:
                    coefficients[i] = 0
            elif "installation" in constrs[i].getAttr("ConstrName").split("_"):
                tokens = constrs[i].getAttr("ConstrName").split("_")
                node, function = tokens[2], tokens[3]
                if params["components assigned"][function] == node:
                    coefficients[i] = self.min_flow_param
                else:
                    coefficients[i] = 0 
            else:
                raise ValueError("No other constraints should contain the path variable")

        # Adds the variable and column.
        self.model.addVar(column = gp.Column(coefficients, constrs), name = path.description + "_flow")


    def compute_optimality_gap(self):
        """
        Computes the dual gap using the upper and lower bounds.
        """
        return self.upper_bound - self.lower_bound/self.lower_bound

    def optimise(self, max_iterations: int = 10, tolerance: float = 0.01):
        """
        Finds schedule that optimises probability of success using column generation
        """
        start = time()
        terminate = False
        
        # Adding initial path.
        logging.info(" Attempting to generate initial path.") if self.verbose > 0 else None
        for service in self.services:
            cgp = self.pricing_problem(service)
            self.add_path_from_model(service, cgp)
            # Updates edges containing dummy node to have arbitrarily high latency so they are not included in future paths.
            for edge in service.graph.links:
                if "dummy" in edge.get_description():
                    edge.latency = inf

        logging.info(" Initialisation completed.\n")

        # Solves restricted master problem using initial points and saves solution.
        logging.info(" BUILDING INITIAL MODEL.\n") if self.verbose > 0 else None
        self.build_initial_model()
        k = 1

        while terminate == False and k < max_iterations:
            logging.info(" Updating service graphs with dual information.\n")
            self.update_duals()
            terminate = True
            self.lower_bound = self.upper_bound

            # Solves the column generation problem for each sub problem.
            for service in self.services:
                logging.info(" SOLVING COLUMN GENERATION FOR {}\n".format(service.description)) if self.verbose > 0 else None
                cg = self.pricing_problem(service)
                self.lower_bound -= cg.objVal
                if cg.objVal < 0:
                    terminate = False
                    logging.info(" New path for service {} has negative reduced cost so adding.\n".format(service.description)) if self.verbose > 0 else None
                    path = self.add_path_from_model(service, cg)
                    path.save_as_dot()
                    self.add_column_from_path(path)
            
            # Updates and resolves RMP.
            logging.info(" SOLVING RMP\n") if self.verbose > 0 else None
            self.rmp()
            k += 1
        
        # Solves with integrality
        for v in self.model.getVars():
            if "assignment" in v.varName or "replication" in v.varName or "isntallation" in v.varName:
                v.setAttr(GRB.Attr.VType, "I")

        self.model.update()
        self.model.optimize()

        logging.info( " No more improving paths found\n.") if self.verbose > 0 else None

        # Prints final solution
        if self.model.status == GRB.OPTIMAL:
            logging.info( " Optimisation terminated successfully\n.") if self.verbose > 0 else None
            logging.info( " Optimality Gap: {}".format(self.compute_optimality_gap()))
            logging.info(' Objective: {}'.format(self.model.objVal)) if self.verbose > 0 else None
            logging.info(' Vars:') if self.verbose > 0 else None
            for v in self.model.getVars():
                if v.x != 0:
                    logging.info(" Variable {}: ".format(v.varName) + str(v.x)) if self.verbose > 0 else None
        else:
            logging.error(" Optimisation Failed - consult .ilp file") if self.verbose > 0 else None
            self.model.computeIIS()
            self.model.write("{}.ilp".format(self.model.getAttr("ModelName")))
            raise ValueError("Optimisation failed")