from multiprocessing.sharedctypes import Value
import this
from matplotlib import use
import numpy as np
from math import log, exp
from scipy import stats
from time import time
from service_class.service import Service
from service_class.graph import service_graph, service_path
from topology.network import Network
from topology.location import Dummy, Node, Switch
from service_class.vnf import VNF
import gurobipy as gp
from gurobipy import GRB
from scipy import optimize
import logging
import json
from math import ceil
import copy

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
    def __init__(self, network: Network, vnfs: list, services: list, verbose: int = 1, logfile: str ='log.txt', max_replicas: int = 3,
                node_availability: float = 0.9999, min_flow_param: int = 10, weights = [1, 100]) -> None:
        """
        Initialises the optimisation problem
        """
        if verbose not in [0, 1, 2]:
            raise ValueError("Invalid verbosity level. Use 0 for no log, 1 for info and 2 for info and debug.")
        self.verbose = verbose
        logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.DEBUG, filemode='w') if self.verbose == True else None

        self.network = network

        # # Adds the dummy node to the network.
        # self.network.add_dummy_node()
        self.vnfs = vnfs
        self.services = services
        self.lower_bound = eps
        self.upper_bound = inf
        self.status = None
        self.runtime = None
        self.model = None
        self.gap = inf
        self.max_replicas = max_replicas
        self.node_availability = node_availability
        self.min_flow_param = min_flow_param
        self.nodes = None
        self.weights = weights
        self.n_nodes_used = None
        self.total_penalties = None

        # Initialises inner approximation for each service
        for service in services:
            service.make_graph(self.vnfs, self.network)

    def pricing_problem(self, service, initial = False, heuristic_solution = None, bigM = 1000):
        """
        Solves the pricing problem for a service and finds the best new path to add to the master problem.
        """
        graph = service.graph
        graph.save_as_dot()
        m = gp.Model(graph.description, env=env)
        links = [l for l in graph.links if not isinstance(l.sink, Dummy) and not isinstance(l.source, Dummy)]

        # Initially we can assume the cost of each edge are all 1 and solve as shortest path to heuristically speed up solution.
        if initial == True:
            w = np.array([1 for l in links])
        else:
            w = np.array([l.cost for l in links])
        
        if heuristic_solution != None:
            bounds = []
            # Fixes assignment edges to be zero if they are the function is not installed in the node in the heuristic solution.
            for l in links:
                if l.assignment_link == True:
                    node, function = graph.get_node_and_function_from_assignment_edge(l, self.vnfs)
                    if node.description not in heuristic_solution[function.description]:
                        bounds.append(0)
                    else:
                        bounds.append(1)
                else:
                    bounds.append(1)
            x = m.addMVar(shape = len(links), ub = bounds, name = [l.get_description() for l in links], vtype = GRB.BINARY)
        else:
            x = m.addMVar(shape = len(links), name = [l.get_description() for l in links], vtype = GRB.BINARY)
        m.update()
        penalty = m.addVar(name = "latencypenalty", vtype = GRB.BINARY)

        # Gets source and sink
        source = graph.get_location_by_description(service.source.description + "_l0")
        sink = graph.get_location_by_description(service.sink.description + "_l{}".format(graph.n_layers - 1))

        # Flow constraints for each node.
        for node in graph.locations:
            if isinstance(node, Dummy): continue
            o_indexes = [links.index(o) for o in graph.outgoing_edge(node) if not isinstance(o.sink, Dummy)]
            i_indexes = [links.index(i) for i in graph.incoming_edge(node) if not isinstance(i.source, Dummy)]
            # 1 edge leaving the source must be active.
            if node == source:
                m.addConstr(gp.quicksum(x[o] for o in o_indexes) == 1, name = "sourceflowout_{}".format(node.description))
                m.addConstr(gp.quicksum(x[i] for i in i_indexes) == 0, name = "sourceflowin_{}".format(node.description))
            # 1 edge entering sink must be active
            elif node == sink:
                m.addConstr(gp.quicksum(x[o] for o in o_indexes) == 0, name = "sinkflowout_{}".format(node.description))
                m.addConstr(gp.quicksum(x[i] for i in i_indexes) == 1, name = "sinkflowin_{}".format(node.description))
            # Flow conservation for every other node.
            else:
                m.addConstr(gp.quicksum(x[o] for o in o_indexes) <= 1, name = "conservationout_{}".format(node.description))
                m.addConstr(gp.quicksum(x[i] for i in i_indexes) <= 1, name = "conservationin_{}".format(node.description))
                m.addConstr(gp.quicksum(x[i] for i in i_indexes) == gp.quicksum(x[o] for o in o_indexes), name = "conservation_{}".format(node.description))
        
        # Adds latency constraint.
        lat = np.array([l.latency for l in links])
        m.addConstr(lat @ x  - bigM * penalty <= service.latency, name = "latency")
        # Gets dual variable associated with constraint that 100% of the flow must be used
        if self.model != None:
            pi = -self.model.getConstrByName("pathflow_{}".format(service.description)).getAttr("Pi")
        else:
            pi = 0

        # Sets objective to reduced cost.
        m.setObjective(self.weights[1] * penalty + pi + w @ x, GRB.MINIMIZE)
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
                    logging.info(" Description: {}, Value: {}".format(links[i].get_description(), vars[i].x)) if self.verbose > 0 else None
            return m
        else:
            logging.error(" Optimisation Failed - consult .ilp") if self.verbose > 0 else None
            m.computeIIS()
            m.write("{}.ilp".format(m.getAttr("ModelName")))
            raise ValueError("Optimisation failed")

    def solve_heuristic(self):
        """
        Heuristically generates a low cost configuration which can be used to generate promising initial paths.
        """
        # Makes list of replica count for required VNF's using availability
        n_replicas = {v.description: 1 for v in self.vnfs}
        function_availability = min([v.availability for v in self.vnfs])
        n_paths_needed = {s.description: 1 for s in self.services}
        nodes = [n for n in self.network.locations if isinstance(n, Node) and not isinstance(n, Dummy)]
        for s in self.services:
            if s.availability != None:
                k = len(s.vnfs)
                # Starting with one, check if availability can be satisfied using one path, else increase 1 until it can be satisfied
                i = 1
                while (1 - (1 - self.node_availability * function_availability)**i)**k < s.availability:
                    i += 1
                # Updates the number of paths needed for each service based on the availability.
                n_paths_needed[s.description] = max(n_paths_needed[s.description], i)
                for v in s.vnfs:
                    # If the required relicas for this service is greater than that previously calculated we update it.
                    n_replicas[v] = max(i, n_replicas[v])

        # if the number of vnfs cannot handle the required throughput, then we increase the replica count so that it can.
        for v in self.vnfs:
            n_instances = ceil(sum([s.throughput for s in self.services if v in s.get_vnfs(self.vnfs)])/v.throughput)
            n_replicas[v.description] = max(n_instances, n_replicas[v.description])

        # Sorts the nodes according to cost
        nodes = [n for _, n in sorted(zip([n.cost for n in nodes], nodes), key=lambda pair: pair[0])]
        # Sorts vnfs according to cores:
        vnfs = [v for _, v in sorted(zip([v.cpu for v in self.vnfs], self.vnfs), key=lambda pair: pair[0], reverse=True)]
        # Start with the cheapest node
        i = 0
        assignments = {f.description: [] for f in vnfs}
        try:
            while any(x != 0 for x in n_replicas.values()) and i < len(nodes):
                current_node = nodes[i]
                cpu_remaining, ram_remaining = nodes[i].cpu, nodes[i].ram
                to_place = vnfs[:]
                while to_place:
                    v = to_place.pop(0)
                    # If it's possible to place it, and an instance still needs to be placed then place it.
                    if v.cpu <= cpu_remaining and v.ram <= ram_remaining and n_replicas[v.description] > 0:
                        assignments[v.description].append(current_node.description)
                        n_replicas[v.description] -= 1
                        cpu_remaining -= v.cpu
                        ram_remaining -= v.ram
                # Move onto next cheapest node.
                i += 1
            logging.info(" HEURISTIC FOUND THE FOLLOWING ASSIGNMENTS {}\n".format(assignments)) if self.verbose > 0 else None
            # Uses heuristic to add paths for each service.
            for service in self.services:
                logging.info(" USING HEURISTIC SOLUTION TO GENERATE COLUMNS FOR {}\n".format(service.description)) if self.verbose > 0 else None
                assignments_to_use = copy.deepcopy(assignments)
                for i in range(n_paths_needed[s.description]):
                    cg = self.pricing_problem(service, initial = True, heuristic_solution = assignments_to_use)
                    if cg.status == GRB.OPTIMAL:
                        # Adds the path if it has a feasible solution.
                        path = self.get_path_from_model(service, cg)
                        service.graph.add_path(path)
                        # Removes the nodes used in the solution from the assignments_to_use dictionary.
                        assignments_used = path.get_params()["components assigned"]
                        logging.info(" Params used: {}".format(path.get_params()))
                        for i in range(len(service.vnfs)):
                            # If the service uses the same VNF twice, then should only remove that VNF once.
                            if assignments_used[i] in assignments_to_use[service.vnfs[i]]:
                                assignments_to_use[service.vnfs[i]].remove(assignments_used[i])
                        logging.info(" Path {}: ".format(path.description) + path.__str__() + " added.\n") if self.verbose > 0 else None
                        
        except IndexError:
            logging.info(" Not possible to assign all services\n".format(assignments)) if self.verbose > 0 else None

    def get_path_from_model(self, service, model):
        """
        Given a service and a column generation model it adds the current path to the service graph.
        """
        graph = service.graph
        link_vars = [v for v in model.getVars() if v.varName != "latencypenalty"]    
        links = [l for l in graph.links if not isinstance(l.sink, Dummy) and not isinstance(l.source, Dummy)]
        # Gets the list of edges used - equal to the edges whose binary variable has value 1.
        used_edges = [links[i] for i in range(len(links)) if link_vars[i].x == 1]

        # Gets the list of used nodes from the used edges.
        used_nodes = []
        for edge in used_edges:
            if edge.source not in used_nodes: used_nodes.append(edge.source)
            if edge.sink not in used_nodes: used_nodes.append(edge.sink)

        # Gets whether latency is violated by path.
        if model.getVarByName("latencypenalty").x == 1:
            penalty_incurred = True
        else:
            penalty_incurred = False

        # Makes the path and adds it to the list of paths for the service graph.
        path = service_path(service.description, used_nodes, used_edges, self.network, service, n_layers = graph.n_layers, latency_violated = penalty_incurred)
        return path

    def add_path_manually(self, service, list_edges):
        """
        Adds a path given a list of strings where each string is the edge description.
        """
        graph = service.graph
        # Gets the list of edges used - equal to the edges whose binary variable has value 1.
        used_edges = [l for l in graph.links if l.get_description() in list_edges]

        # Gets the list of used nodes from the used edges.
        used_nodes = []
        for edge in used_edges:
            if edge.source not in used_nodes: used_nodes.append(edge.source)
            if edge.sink not in used_nodes: used_nodes.append(edge.sink)

        # Makes the path and adds it to the list of paths for the service graph.
        path = service_path(service.description, used_nodes, used_edges, self.network, service, n_layers = graph.n_layers)
        return path

    def build_initial_model(self):
        """
        Builds and solves the restricted master problem given the initial approximation points.
        """
        self.model = gp.Model(self.network.description, env=env)
        nodes = [n for n in self.network.locations if isinstance(n, Node) == True]
        self.nodes = nodes

        vnfs = set()
        # Also makes set of components for all services so that no duplicate components are considered
        for service in self.services:
            for f in service.get_vnfs(self.vnfs):
                vnfs.add(f)
        self.vnfs = list(vnfs)

        # Gets set of edges - not including reverse edges since edges are bidirectional.
        edges = []
        for edge in self.network.links:
            if edge.get_description() and edge.get_opposing_edge_description() not in [e.get_description() for e in edges]:
                edges.append(edge)

        # Adds variables, for LP relaxation makes variables continuous.
        for node in nodes:
            # Adds variable that says whether a node is used.
            self.model.addVar(lb = 0, ub = 1, vtype=GRB.CONTINUOUS, obj = node.cost, name= node.description + "_used")
            for function in vnfs:
                # Adds variable that says a function is installed on a node.
                self.model.addVar(lb = 0, ub = 1, vtype=GRB.CONTINUOUS, obj = node.cost, name= node.description + "_" + function.description + "_assignment")
 
        for service in self.services:
            # Adds availability penalty variable.
            self.model.addVar(lb = 0, ub = 1, vtype = GRB.CONTINUOUS, name = service.description + "_availabilitypenalty")
            # Adds throughput penalty variable.
            self.model.addVar(lb = 0, ub = 1, vtype = GRB.CONTINUOUS, name = service.description + "_throughputpenalty")
            for function in service.get_vnfs(self.vnfs):
                for node in self.nodes:
                    # Adds service installation variables.
                    self.model.addVar(lb = 0, ub = 1, vtype=GRB.CONTINUOUS, name= service.description + "_" + node.description + "_" + function.description + "_installation")
                for i in range(1, self.max_replicas + 1):
                    # Adds service replication variables.
                    self.model.addVar(lb = 0, ub = 1, vtype=GRB.CONTINUOUS, name= service.description + "_" + function.description + "_" + str(i) + "_replication")
            for path in service.graph.paths:
                # Adds service path flow variables.
                self.model.addVar(vtype=GRB.CONTINUOUS, name = path.description + "_flow")

        self.model.update()
        # Adds a constraint that says if any function is hosted on a node, then the node is considered used.
        for node in nodes:
            for function in vnfs:
                node_var = self.model.getVarByName(node.description + "_used")
                function_var = self.model.getVarByName(node.description + "_" + function.description + "_assignment")
                self.model.addConstr(node_var >= function_var, name = node.description + "_" + function.description + "_used")

        # Adds constraint that the sum of CPU and RAM of the functions does not exceed the capacity of the nodes.
        for node in nodes:
            vars_used = []
            cpus = []
            rams = []
            for function in vnfs:
                vars_used.append(self.model.getVarByName(node.description + "_" + function.description + "_assignment"))
                cpus.append(function.cpu)
                rams.append(function.ram)
            self.model.addConstr(gp.quicksum(cpus[i] * vars_used[i] for i in range(len(vars_used))) <= node.cpu, name = "cpu_{}".format(node.description))
            self.model.addConstr(gp.quicksum(rams[i] * vars_used[i] for i in range(len(vars_used))) <= node.ram, name = "ram_{}".format(node.description))

        # Adds constraint that says the sum of all flows through each edge must not exceed the bandwidth capacity.
        for edge in edges:
            links_used, vars_used = [], []
            for s in self.services:
                for p in s.graph.paths:
                    # If the edge is used in the path.
                    if edge.get_description() in p.get_params()["times traversed"].keys():
                        links_used.append(p.get_params()["times traversed"][edge.get_description()])
                        vars_used.append(self.model.getVarByName(p.description + "_flow"))
            self.model.addConstr(gp.quicksum(service.throughput * links_used[i] * vars_used[i] for i in range(len(vars_used))) <= edge.bandwidth, name = "bandwidth_{}".format(edge.get_description()))

        # Adds a constraint that says for each service, 100% of the flow must be routed.
        for service in self.services:
            vars_used = [self.model.getVarByName(p.description + "_flow") for p in service.graph.paths]
            penalty_var = self.model.getVarByName(service.description + "_throughputpenalty")
            self.model.addConstr(gp.quicksum(vars_used) + penalty_var == 1, name = "pathflow_{}".format(service.description))

        # Adds a constraint that says, if a vnf is not assigned to a node, then every path in that service, that considers the vnf to be installed on that node must have zero flow.
        for service in self.services:
            for node in self.nodes:
                for function in service.get_vnfs(self.vnfs):
                    flow_vars_used = []
                    # Gets the path flow variables which assume the VNF to be hosted on the node.
                    for path in service.graph.paths:
                        if path.check_if_using_assignment(function, node) == True:
                            flow_vars_used.append(self.model.getVarByName(path.description + "_flow"))
                    # Gets the assignment variables and parameters.
                    assignment_var = self.model.getVarByName(node.description + "_" + function.description + "_assignment")
                    self.model.addConstr(gp.quicksum(flow_vars_used[i] for i in range(len(flow_vars_used))) <= assignment_var, name="assignment_{}_{}_{}".format(service.description, node.description, function.description))
        
        # Adds a constraint that says the sum of flows through any instance of a VNF must not exceed it's processing capacity.
        for node in nodes:
            for function in vnfs:
                flow_vars_used, throughput_params = [], []
                for service in self.services:
                    if function in service.get_vnfs(self.vnfs):
                        for path in service.graph.paths:
                            if path.check_if_using_assignment(function, node) == True:
                                flow_vars_used.append(self.model.getVarByName(path.description + "_flow"))
                                throughput_params.append(service.throughput)
                # Gets the assignment variables and parameters.
                assignment_var = self.model.getVarByName(node.description + "_" + function.description + "_assignment")
                self.model.addConstr(gp.quicksum(flow_vars_used[i] * throughput_params[i] for i in range(len(flow_vars_used))) <= assignment_var * function.throughput, name="throughput_{}_{}".format(node.description, function.description))
        
        # Adds a constraint that forces the installation variable of a service to be 1 if at least one path using that node is used to route flow.
        for service in self.services:
            for node in self.nodes:
                for function in service.get_vnfs(self.vnfs):
                    flow_vars_used = []
                    # Gets the path flow variables which assume the VNF to be hosted on the node.
                    for path in service.graph.paths:
                        if path.check_if_using_assignment(function, node) == True:
                            flow_vars_used.append(self.model.getVarByName(path.description + "_flow"))
                    installation_var = self.model.getVarByName(service.description + "_" + node.description + "_" + function.description + "_installation")
                    self.model.addConstr(self.min_flow_param * gp.quicksum(flow_vars_used[i] for i in range(len(flow_vars_used))) >= installation_var, name="installation_{}_{}_{}".format(service.description, node.description, function.description))
             
        # Adds a constraint that forces one of the replication variables to take a value of one.
        for service in self.services:
            for function in service.get_vnfs(self.vnfs):
                vars_used = [self.model.getVarByName(service.description+ "_" + function.description + "_" + str(i) + "_replication") for i in range(1, self.max_replicas + 1)]
                self.model.addConstr(gp.quicksum(vars_used) == 1, name = "replication_{}_{}".format(service.description, function.description))

        # Adds a constraint that constrains the number of replicas to be equal to the number of different nodes hosting that function across the service paths.
        for service in self.services:
            for function in service.get_vnfs(self.vnfs):
                #dummy_installation = self.model.getVarByName(service.description + "_" + dummy_node.description + "_" + function.description + "_installation")
                replication_vars_used = [self.model.getVarByName(service.description+ "_" + function.description + "_" + str(i) + "_replication") for i in range(1, self.max_replicas + 1)]
                installation_var_used = [self.model.getVarByName(service.description + "_" + n.description + "_" + function.description + "_installation") for n in nodes if isinstance(n, Dummy) == False]
                params = [i for i in range(1, self.max_replicas + 1)]
                self.model.addConstr(gp.quicksum(params[i] * replication_vars_used[i] for i in range(len(replication_vars_used))) <= gp.quicksum(installation_var_used), name = "nreplicas_{}_{}".format(service.description, function.description))

        # Adds availability constraints for each service.
        for service in self.services:
            if service.availability != None:
                rhs = log(service.availability)
                vars_used, params = [], [], 
                for function in service.get_vnfs(self.vnfs):
                    for i in range(1, self.max_replicas + 1):
                        vars_used.append(self.model.getVarByName(service.description+ "_" + function.description + "_" + str(i) + "_replication"))
                        params.append(log(1 - (1 - self.node_availability * function.availability) ** i))
                penalty = self.model.getVarByName(service.description + "_availabilitypenalty")
                self.model.addConstr(gp.quicksum(params[i] * vars_used[i] for i in range(len(vars_used))) + penalty >= rhs, name = "availability_{}".format(service.description))

        # Updates and optimises the model.
        self.model.update()
        node_used_vars = [v for v in self.model.getVars() if "_used" in v.varName]
        av_pens = [v for v in self.model.getVars() if "_availabilitypenalty" in v.varName]
        tp_pens = [v for v in self.model.getVars() if "_throughputpenalty" in v.varName]
        lt_pens, path_vars = [], []
        for s in self.services:
            for p in s.graph.paths:
                if p.latency_violated == True:
                    path_vars.append(self.model.getVarByName(p.description + "_flow"))
                    lt_pens.append(1)
        
        # Makes the objective the combination of nodes used, and SLA violation costs.
        self.model.setObjective(self.weights[0] * gp.quicksum(node_used_vars[i] for i in range(len(node_used_vars))) + 
                                self.weights[1] * (gp.quicksum(av_pens[i] for i in range(len(av_pens))) + gp.quicksum(tp_pens[i] for i in range(len(tp_pens)))
                                + gp.quicksum(lt_pens[i] * path_vars[i] for i in range(len(lt_pens)))))
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
                if c.getAttr("Pi") != 0:
                    logging.info(" Constr {}: ".format(c.getAttr("ConstrName")) + str(c.getAttr("Pi"))) if self.verbose > 0 else None
        else:
            logging.error(" Optimisation Failed - consult .ilp file") if self.verbose > 0 else None
            self.model.computeIIS()
            self.model.write("{}.ilp".format(self.model.getAttr("ModelName")))
            raise ValueError("Optimisation failed")
    
    def rmp(self, integrality = False):
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
            if integrality == False:
                logging.info(" Dual Solution:") if self.verbose > 0 else None
                for c in self.model.getConstrs():
                    if c.getAttr("Pi") != 0:
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
                    # To avoid adding arbitrary links due to zero cost edges we set the cost to a small value
                    if -self.model.getConstrByName("bandwidth_{}".format(e_o.get_description())).getAttr("Pi") == 0:
                        e.cost = 1e-6
                    else:
                        e.cost = -self.model.getConstrByName("bandwidth_{}".format(e_o.get_description())).getAttr("Pi")
                else:
                    raise KeyError("Bandwidth constraint not found for edge.")

            # Updates the assignment edge cost using the duals associated with the assignment constraints.
            assignment_edges = [e for e in graph.links if e.assignment_link == True]
            for e in assignment_edges:
                node, function = graph.get_node_and_function_from_assignment_edge(e, self.vnfs)
                constr1 = self.model.getConstrByName("assignment_{}_{}_{}".format(service.description, node.description, function.description))
                constr2 = self.model.getConstrByName("throughput_{}_{}".format(node.description, function.description))
                constr3 = self.model.getConstrByName("installation_{}_{}_{}".format(service.description, node.description, function.description))
                if constr1 != None and constr2 != None and constr3 != None:
                    # To avoid adding arbitrary links due to zero cost edges we set the cost to a small value
                    if -(constr1.getAttr("Pi") + constr2.getAttr("Pi") + self.min_flow_param / service.throughput * constr3.getAttr("Pi")) == 0:
                        e.cost = 1e-6
                    else:
                        e.cost = -(constr1.getAttr("Pi") + service.throughput * constr2.getAttr("Pi") + self.min_flow_param * constr3.getAttr("Pi"))
                else:
                    raise KeyError("Constraint not found for this service, node, function combination.")
    
    def add_column_from_path(self, path):
        """
        Adds a new path variable with column of coefficients taken from the pricing problem.
        """
        service = path.service
        params = path.get_params()
        constrs = self.model.getConstrs()
        coefficients = np.zeros(len(constrs))

        # Gets the coefficients in the column associated with the new path variable in each constraint.
        for i in range(len(constrs)):
            if "bandwidth" in constrs[i].getAttr("ConstrName"):
                # For bandwidth constraints this is the number of times the link is used
                edge = self.network.get_link_by_description(constrs[i].getAttr("ConstrName").split("_")[-1])
                if edge.get_description() in params["times traversed"].keys():
                    coefficients[i] = params["times traversed"][edge.get_description()]
                else:
                    coefficients[i] = 0
            elif "pathflow_{}".format(service.description) in constrs[i].getAttr("ConstrName"):
                # This is just 1 since we sum all the path flow variables.
                coefficients[i] = 1
            elif "assignment_{}".format(service.description) in constrs[i].getAttr("ConstrName"):
                # This is 1 if the path has the function hosted on a particular node.
                tokens = constrs[i].getAttr("ConstrName").split("_")
                node, function = tokens[2], tokens[3]
                if function in path.service.vnfs:
                    if path.check_if_using_assignment(function, node) == True:
                        coefficients[i] = 1
                else:
                    coefficients[i] = 0
            elif "throughput" in constrs[i].getAttr("ConstrName"):
                tokens = constrs[i].getAttr("ConstrName").split("_")
                node, function = tokens[1], tokens[2]
                if function in path.service.vnfs:
                    if path.check_if_using_assignment(function, node) == True:
                        coefficients[i] = service.throughput
                else:
                    coefficients[i] = 0
            elif "installation_{}".format(service.description) in constrs[i].getAttr("ConstrName"):
                tokens = constrs[i].getAttr("ConstrName").split("_")
                node, function = tokens[2], tokens[3]
                if function in path.service.vnfs:
                    if path.check_if_using_assignment(function, node) == True:
                        coefficients[i] = self.min_flow_param
                else:
                    coefficients[i] = 0
            else:
                coefficients[i] = 0

        # Adds the variable and column.
        if path.latency_violated == True:    
            self.model.addVar(obj = self.weights[1] ,column = gp.Column(coefficients, constrs), name = path.description + "_flow")
        else:
            self.model.addVar(column = gp.Column(coefficients, constrs), name = path.description + "_flow")

    def compute_optimality_gap(self):
        """
        Computes the dual gap using the upper and lower bounds.
        """
        return self.upper_bound - self.lower_bound/self.lower_bound

    def optimise(self, max_iterations: int = 1000, tolerance: float = 0.05, use_heuristic = False):
        """
        Finds schedule that optimises probability of success using column generation
        """
        terminate = False

        if use_heuristic == True:
            self.solve_heuristic()

        # Solves shortest path for each service and adds the path for heuristic value.
        for service in self.services:
            logging.info(" SOLVING INITIAL COLUMN GENERATION FOR {}\n".format(service.description)) if self.verbose > 0 else None
            cg = self.pricing_problem(service, initial = True)
            path = self.get_path_from_model(service, cg)
            service.graph.add_path(path)
            logging.info(" Path {}: ".format(path.description) + path.__str__() + " added.") if self.verbose > 0 else None
            logging.info(" Params used: {}\n".format(path.get_params()))

        # Solves restricted master problem using initial points and saves solution.
        logging.info(" BUILDING INITIAL MODEL.\n") if self.verbose > 0 else None
        self.build_initial_model()
        self.upper_bound = self.model.objVal
        start = time()
        k = 1

        while terminate == False and k <= max_iterations:
            logging.info(" Updating service graphs with dual information.\n")
            self.update_duals()
            terminate = True
            self.upper_bound, self.lower_bound = self.model.objVal, self.model.objVal
            # Solves the column generation problem for each sub problem.
            for service in self.services:
                #if service.status == False:
                logging.info(" SOLVING COLUMN GENERATION FOR {}\n".format(service.description)) if self.verbose > 0 else None
                cg = self.pricing_problem(service)
                self.lower_bound -= cg.objVal
                if cg.objVal < 0:
                    terminate = False
                    logging.info(" New path for service {} has negative reduced cost so adding.".format(service.description)) if self.verbose > 0 else None
                    path = self.get_path_from_model(service, cg)
                    # Checks that the path is unique (i.e. not already in set of paths.)
                    if not any([p.check_if_same(path) for p in service.graph.paths]):
                        service.graph.add_path(path)
                        self.add_column_from_path(path)
                        logging.info(" Path {}: ".format(path.description) + path.__str__() + " added.") if self.verbose > 0 else None
                        logging.info(" Params used: {}\n".format(path.get_params()))
                    else:
                        logging.info(" Path not unique so not adding.") if self.verbose > 0 else None
                else:
                    service.status = True
                    logging.info(" Not an improving path.") if self.verbose > 0 else None
            self.gap = self.compute_optimality_gap()


            logging.info(" SOLVING RMP\n") if self.verbose > 0 else None
            self.rmp()
            k += 1

        logging.info( " No more improving paths found.\n.") if self.verbose > 0 else None

        # Sets lower bound to the solution from the LRMP
        self.lower_bound = self.model.objVal

        # Solves with integrality
        for v in self.model.getVars():
            if "assignment" in v.varName or "replication" in v.varName or "installation" in v.varName:
                v.setAttr(GRB.Attr.VType, "I")

        logging.info( " Solving with integrality constraints.\n.") if self.verbose > 0 else None
        self.rmp(integrality=True)
        self.upper_bound = self.model.objVal
        
        # Prints final solution
        if self.model.status == GRB.OPTIMAL:
            self.runtime = time() - start
            self.gap = self.compute_optimality_gap()
            self.status = True
            self.parse_solution()
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
            self.status = False
    
    def parse_solution(self):
        """"
        If the optimisation was successful, it parses the solution and updates the relevant objects with the information.
        """
        nodes = [n for n in self.network.locations if isinstance(n, Node) == True]
        # Adds the assignment of VNF's to the node
        nodes_used = []
        for node in nodes:
            assignments = []
            for function in self.vnfs:
                var = self.model.getVarByName(node.description + "_" + function.description + "_assignment")
                if var.x == 1:
                    assignments.append(function.description)
                    if node not in nodes_used:
                        nodes_used.append(node)
            node.assignments = assignments
            self.n_nodes_used = len(nodes_used)

        total_penalties = 0
        for service in self.services:
            sla_violations = {}
            tp = self.model.getVarByName(service.description + "_throughputpenalty").x
            av = self.model.getVarByName(service.description + "_availabilitypenalty").x
            total_penalties += tp
            total_penalties += av
            sla_violations["throughput"] = tp
            sla_violations["availability"] = av
            sla_violations["latency"] = {}
            for path in service.graph.paths:
                flow = self.model.getVarByName(name = path.description + "_flow").x
                path.flow = flow
                if flow != 0 and path.latency_violated == True:
                    sla_violations["latency"][path.description] = flow
                    total_penalties += flow
            service.sla_violations = sla_violations
        self.total_penalties = total_penalties
                
    def to_json(self) -> dict:
        """
        Returns a json dictionary describing the model.
        """
        to_return = {}
        to_return["status"] = self.status
        to_return["objective"] = self.model.objVal
        to_return["number of nodes used"] = self.n_nodes_used
        to_return["total penalties"] = self.total_penalties 
        to_return["runtime"] = self.runtime
        to_return["network"] = self.network.to_json()
        to_return["vnfs"] = [v.to_json() for v in self.vnfs]
        to_return["services"] = [s.to_json() for s in self.services]
        return to_return

    def save_as_json(self, filename = None):
        """
        saves the network as a JSON to filename.json
        """
        to_dump = self.to_json()
        if filename != None:
            if filename[-5:] != ".json":
                filename = filename + ".json"
        else:
            filename = self.description + ".json"

        with open(filename, 'w') as fp:
            json.dump(to_dump, fp, indent=4, separators=(", ", ": "))

    
