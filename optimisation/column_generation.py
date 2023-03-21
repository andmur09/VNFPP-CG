from multiprocessing.sharedctypes import Value
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
                node_availability: float = 0.9999, min_flow_param: int = 100) -> None:
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
        self.vnfs = vnfs
        self.nodes = None
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
        
        # Initialises inner approximation for each service
        for service in services:
            service.make_graph(self.vnfs, self.network)

    def pricing_problem(self, service, initial = False):
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
        x = m.addMVar(shape = len(links), name = [l.get_description() for l in links], vtype = GRB.BINARY)
        m.update()

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
        # m.addConstr(lat @ x <= service.latency, name = "latency")
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
                    logging.info(" Description: {}, Value: {}".format(links[i].get_description(), vars[i].x)) if self.verbose > 0 else None
            return m
        else:
            logging.error(" Optimisation Failed - consult .ilp") if self.verbose > 0 else None
            m.computeIIS()
            m.write("{}.ilp".format(m.getAttr("ModelName")))
            raise ValueError("Optimisation failed")
    
    def find_cover_cut(self, function):
        """
        For a given LP solution, uses the requiredinstances constraint for each VNF to add cover cuts if found. This problem finds a cover, which must be satisfied by
        the integer solution and is not satisfied by the LP solution.
        """
        nodes = [n for n in self.network.locations if isinstance(n, Node) == True]
        self.nodes = nodes
        bound = sum([s.throughput for s in self.services if function.description in s.vnfs])
        m = gp.Model(function.description + "_cover", env=env)
        lp_vals = []
        coeffs = []
        for node in nodes:
            if isinstance(node, Dummy):
                # Gets the LP value associated with the variable.
                lp_vals.append(self.model.getVarByName(node.description + "_" + function.description + "_assignment").x)
                # Adds the binary assignment variable.
                m.addVar(lb = 0, ub = 1, vtype=GRB.BINARY, name= node.description + "_" + function.description + "_assignment")
                # Gets the coefficients associated with the cover constraint.
                coeffs.append(bound)
            else:
                max_instances = min(8, node.cpu // function.cpu)
                for i in range(1, max_instances + 1):
                    # Gets the LP value associated with the variable.
                    lp_vals.append(self.model.getVarByName(node.description + "_" + function.description + "_" + str(i) + "_assignment").x)
                    # Adds the binary assignment variable.
                    m.addVar(lb = 0, ub = 1, vtype=GRB.BINARY, name= node.description + "_" + function.description + "_" + str(i) + "_assignment")
                    # Gets the coefficients associated with the cover constraint.
                    coeffs.append(i * function.throughput)
        m.update()
        vars = m.getVars()
        # Adds the cover constraint. If all assignment variables taking a value of 1, were to be used in the MIP, this constraint would be violated. As such,
        # For this constraint to be satisfied at least one of the other variables (those with current value 0) must be active.
        m.addConstr(gp.quicksum(coeffs[i]*vars[i] for i in range(len(coeffs))) <= bound - 1, name = "cover")

        for node in nodes:
            vars_used = []
            for v in vars:
                node_name = v.varName.split("_")[0]
                if node_name == node.description:
                    vars_used.append(v)
            # Adds the constraint that says that only one assignment var for each function/node pair must be active.
            m.addConstr(gp.quicksum(vars_used[i] for i in range(len(vars_used))) <= 1, name = node.description + "_assignment")

        # This objective finds the value of all LP variables whose integer solution are zero. This should be less than 1 for the cover to be satisfied by the fractional solution.
        m.setObjective(gp.quicksum(lp_vals[i] * (1 - vars[i]) for i in range(len(vars))), GRB.MINIMIZE)
        # Updates and optimizes.
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
                    logging.info(" Description: {}, Value: {}".format(vars[i].varName, vars[i].x)) if self.verbose > 0 else None
            return m
        else:
            logging.error(" Optimisation Failed - consult .ilp") if self.verbose > 0 else None
            m.computeIIS()
            m.write("{}.ilp".format(m.getAttr("ModelName")))
            raise ValueError("Optimisation failed")   

    def find_cover_cut_node(self, node):
        """
        For a given LP solution, uses the requiredinstances constraint for each VNF to add cover cuts if found. This problem finds a cover, which must be satisfied by
        the integer solution and is not satisfied by the LP solution.
        """
        
        m = gp.Model(node.description + "_cover", env=env)
        lp_vals = []
        coeffs = []
        for function in self.vnfs:
            if function.throughput == None:
                # Gets the LP value associated with the variable.
                lp_vals.append(self.model.getVarByName(node.description + "_" + function.description + "_assignment").x)
                # Adds the binary assignment variable.
                m.addVar(lb = 0, ub = 1, vtype=GRB.BINARY, name= node.description + "_" + function.description + "_assignment")
                # Gets the coefficients associated with the cover constraint.
                coeffs.append(function.cpu)
            else:
                max_instances = min(8, node.cpu // function.cpu)
                for i in range(1, max_instances + 1):
                    # Gets the LP value associated with the variable.
                    lp_vals.append(self.model.getVarByName(node.description + "_" + function.description + "_" + str(i) + "_assignment").x)
                    # Adds the binary assignment variable.
                    m.addVar(lb = 0, ub = 1, vtype=GRB.BINARY, name= node.description + "_" + function.description + "_" + str(i) + "_assignment")
                    # Gets the coefficients associated with the cover constraint.
                    coeffs.append(i * function.cpu)
        m.update()
        vars = m.getVars()
        # Adds the cover constraint. If all assignment variables taking a value of 1, were to be used in the MIP, this constraint would be violated. As such,
        # For this constraint to be satisfied at least one of the other variables (those with current value 0) must be active.
        m.addConstr(gp.quicksum(coeffs[i]*vars[i] for i in range(len(coeffs))) >= node.cpu + 1, name = "cover")

        # This objective finds the value of all LP variables whose integer solution are zero. This should be less than 1 for the cover to be satisfied by the fractional solution.
        m.setObjective(gp.quicksum((1 - lp_vals[i]) * vars[i] for i in range(len(vars))), GRB.MINIMIZE)
        # Updates and optimizes.
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
                    logging.info(" Description: {}, Value: {}".format(vars[i].varName, vars[i].x)) if self.verbose > 0 else None
            return m
        else:
            logging.error(" Optimisation Failed - consult .ilp") if self.verbose > 0 else None
            m.computeIIS()
            m.write("{}.ilp".format(m.getAttr("ModelName")))
            raise ValueError("Optimisation failed")   

    def get_path_from_model(self, service, model):
        """
        Given a service and a column generation model it adds the current path to the service graph.
        """
        graph = service.graph
        vars = model.getVars()        
        links = [l for l in graph.links if not isinstance(l.sink, Dummy) and not isinstance(l.source, Dummy)]
        # Gets the list of edges used - equal to the edges whose binary variable has value 1.
        used_edges = [links[i] for i in range(len(links)) if vars[i].x == 1]

        # Gets the list of used nodes from the used edges.
        used_nodes = []
        for edge in used_edges:
            if edge.source not in used_nodes: used_nodes.append(edge.source)
            if edge.sink not in used_nodes: used_nodes.append(edge.sink)

        # Makes the path and adds it to the list of paths for the service graph.
        path = service_path(service.description, used_nodes, used_edges, self.network, service, n_layers = graph.n_layers)
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

    # def greedy_initialisation(self):
    #     """
    #     Finds a valid configuration for each service
    #     """
    #     components_assigned = {}
    #     nodes = [n for n in self.network.locations if isinstance(n, Node) == True]
    #     costs = [n.cost for n in nodes]
    #     # Sorts nodes in terms of cost.
    #     sorted_nodes = [n for _, n in sorted(zip([n.cost for n in nodes], nodes), key=lambda pair: pair[0])]
    #     # Sorts vnfs in terms of cpu.
    #     sorted_vnfs = [f for _, f in sorted(zip([f.cpu for f in self.vnfs], self.vnfs), key=lambda pair: pair[0])]
    #     number_replicas = {}
    #     # Counts number of replicas required by each service
    #     for function in self.vnfs:
    #         throughput_required = sum([s.throughput for s in self.services uf ])


    #     # While there are nodes we haven't explored
    #     while sorted_nodes:
    #         # If there are no VNF's to assign we can break out of the loop.
    #         if not sorted_vnfs:
    #             break
    #         # Assigns current node to the cheapest
    #         current_node = sorted_nodes.pop(0)
    #         # Initialises the remaining ram and cpu capacity of the node
    #         remaining_cpu, remaining_ram = current_node.cpu, current_node.ram
    #         # while there is still a vnf to assign and it is possible to host it on the current node, assign it.
    #         while sorted_vnfs and remaining_ram >= sorted_vnfs[0].ram and remaining_cpu >= sorted_vnfs[0].cpu:
    #             to_add = sorted_vnfs.pop(0)
    #             components_assigned[to_add.description] = current_node.description
    #             remaining_cpu -= to_add.cpu
    #             remaining_ram -= to_add.ram

    #     # At this stage not all services are satisfied by availability etc.



    #     for node in sorted_nodes:
    #         remaining_ram, remaining_cpu = node.ram, node.cpu
    #         current_vnf = sorted_vnfs.pop(0)
    #         while remaining_ram >= current_vnf.ram and remaining_cpu >= current_vnf.cpu:
    #             components_assigned[current_vnf.description] = node.description
    #             remaining_ram -= current_vnf.ram
    #             remaining_cpu -= current_vnf.cpu
    #             current_vnf = sorted_vnfs.pop(0)
            
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
            for f in service.get_vnfs(self.vnfs):
                vnfs.add(f)
        self.vnfs = list(vnfs)

        # Gets set of edges - not including reverse edges since edges are bidirectional.
        edges = []
        for edge in self.network.links:
            if edge.get_description() and edge.get_opposing_edge_description() not in [e.get_description() for e in edges]:
                edges.append(edge)

        # Adds variables, for LP relaxation makes variables continuous.
        # Assignment variables.
        for node in nodes:
            for function in vnfs:
                # If the VNF has no capacity on how much traffic it can process or if the node is a dummy node we add one variable since an unlimited number of vnfs can be hosted on the node.
                if function.throughput == None or isinstance(node, Dummy) == True:
                    self.model.addVar(lb = 0, ub = 1, vtype=GRB.CONTINUOUS, obj = node.cost, name= node.description + "_" + function.description + "_assignment")
                else:
                    # Given the number of cores needed per instance, and the number of cores in the node, calculates the max number of replicas of the VNFs
                    # that can be hosted on the node.
                    max_instances = min(8, node.cpu // function.cpu)
                    for i in range(1, max_instances + 1):
                        self.model.addVar(lb = 0, ub = 1, vtype=GRB.CONTINUOUS, obj = node.cost, name= node.description + "_" + function.description + "_" + str(i) + "_assignment")

        # Path variables.
        for service in self.services:
            for path in service.graph.paths:
                self.model.addVar(vtype=GRB.CONTINUOUS, name = path.description + "_flow")
        
        # Service installation variables
        for service in self.services:
            for node in nodes:
                for function in service.get_vnfs(self.vnfs):
                    self.model.addVar(lb = 0, ub = 1, vtype=GRB.CONTINUOUS, name= service.description + "_" + node.description + "_" + function.description + "_installation")

        # Service replication variables
        for service in self.services:
            for function in service.get_vnfs(self.vnfs):
                for i in range(1, self.max_replicas + 1):
                    self.model.addVar(lb = 0, ub = 1, vtype=GRB.CONTINUOUS, name= service.description + "_" + function.description + "_" + str(i) + "_replication")

        self.model.update()

        # Adds constraint that the sum of CPU and RAM of the functions does not exceed the capacity of the nodes.
        for node in nodes:
            vars_used = []
            cpus = []
            rams = []
            for function in vnfs:
                if isinstance(node, Dummy) == True:
                    vars_used.append(self.model.getVarByName(node.description + "_" + function.description + "_assignment"))
                    cpus.append(function.cpu)
                    rams.append(function.ram)
                elif function.throughput == None:
                    vars_used.append(self.model.getVarByName(node.description + "_" + function.description + "_assignment"))
                    cpus.append(function.cpu)
                    rams.append(function.ram)
                else:
                    max_instances = min(8, node.cpu // function.cpu)
                    for i in range(1, max_instances + 1):
                        vars_used.append(self.model.getVarByName(node.description + "_" + function.description + "_" + str(i) + "_assignment"))
                        cpus.append(i * function.cpu)
                        rams.append(i * function.ram)
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
            self.model.addConstr(gp.quicksum(links_used[i] * vars_used[i] for i in range(len(vars_used))) <= edge.bandwidth, name = "bandwidth_{}".format(edge.get_description()))

        # Adds a constraint that says for each service, 100% of the flow must be routed.
        for service in self.services:
            vars_used = [self.model.getVarByName(p.description + "_flow") for p in service.graph.paths]
            self.model.addConstr(gp.quicksum(vars_used) == service.throughput, name = "pathflow_{}".format(service.description))

        # Adds a constraint that says, if a vnf is not assigned to a node, then every path that considers the vnf to be installed on that node must have zero flow.
        for node in nodes:
            for function in vnfs:
                flow_vars_used, assignment_vars_used, throughput_params = [], [], []
                # Gets the flow variables that use the node, function combination.
                for service in self.services:
                    for path in service.graph.paths:
                        assignments = path.get_params()["components assigned"]
                        if function in service.get_vnfs(self.vnfs) and assignments[function.description] == node.description:
                            flow_vars_used.append(self.model.getVarByName(path.description + "_flow"))
                # Gets the assignment variables and parameters.
                if function.throughput == None or isinstance(node, Dummy):
                    assignment_vars_used.append(self.model.getVarByName(node.description + "_" + function.description + "_assignment"))
                    throughput_params.append(sum([s.throughput for s in self.services]))
                else:
                    max_instances = min(8, node.cpu // function.cpu)
                    for i in range(1, max_instances + 1):
                        assignment_vars_used.append(self.model.getVarByName(node.description + "_" + function.description + "_" + str(i) + "_assignment"))
                        throughput_params.append(i * function.throughput)
                self.model.addConstr(gp.quicksum(flow_vars_used[i] for i in range(len(flow_vars_used))) <= gp.quicksum(throughput_params[j] * assignment_vars_used[j] for j in range(len(assignment_vars_used))), name="assignment_{}_{}".format(node.description, function.description))
                
        # Adds a constraint that forces at most one of the assignment variables to take a value of 1
        for node in nodes:
            for function in vnfs:
                if function.throughput != None and isinstance(node, Dummy) == False:
                    max_instances = min(8, node.cpu // function.cpu)
                    vars_used = [self.model.getVarByName(node.description + "_" + function.description + "_" + str(i) + "_assignment") for i in range(1, max_instances + 1)]
                    self.model.addConstr(gp.quicksum(vars_used[i] for i in range(len(vars_used))) <= 1, name="scaling_{}_{}".format(node.description, function.description))

        # # Adds a constraint that forces the throughput capacity of vnfs used to be at least the total throughput required for the services using the vnfs.
        # for function in vnfs:
        #     throughput_required = sum([s.throughput for s in self.services if function.description in s.vnfs])
        #     if function.throughput == None:
        #         vars = []
        #         for node in nodes:
        #             vars.append(self.model.getVarByName(name= node.description + "_" + function.description + "_assignment"))
        #         self.model.addConstr(gp.quicksum(vars[i] for i in range(len(vars))) >= 1, name="requiredinstances_{}".format(function.description))
        #     else:
        #         # In case the required throughput is less than the throughput processing capability of one instance of the VNF,
        #         # then we just make the max the throughput capability of the VNF. This has the same effect as above, the installation
        #         # variables must be at least 1.
        #         limit = max(throughput_required, function.throughput)
        #         vars, params = [], []
        #         for node in nodes:
        #             if isinstance(node, Dummy):
        #                 vars.append(self.model.getVarByName(name= node.description + "_" + function.description + "_assignment"))
        #                 params.append(limit)
        #             else:
        #                 max_instances = min(8, node.cpu // function.cpu)
        #                 for i in range(1, max_instances + 1):
        #                     vars.append(self.model.getVarByName(node.description + "_" + function.description + "_" + str(i) + "_assignment"))
        #                     params.append(i * function.throughput)
        #         self.model.addConstr(gp.quicksum(params[i] * vars[i] for i in range(len(vars))) >= limit, name="requiredinstances_{}".format(function.description))

        # Adds a constraint that forces the installation variable of a service to be 1 if at least one path using that node is used to route flow.
        for service in self.services:
            for node in nodes:
                for function in service.get_vnfs(self.vnfs):
                    installation_var_used = self.model.getVarByName(service.description + "_" + node.description + "_" + function.description + "_installation")
                    flow_vars_used = [self.model.getVarByName(p.description + "_flow") for p in service.graph.paths]
                    assignment_params = [1 if p.get_params()["components assigned"][function.description] == node.description else 0 for p in service.graph.paths]
                    self.model.addConstr(self.min_flow_param/service.throughput * gp.quicksum(assignment_params[i] * flow_vars_used[i] for i in range(len(service.graph.paths))) >= installation_var_used, name="installation_{}_{}_{}".format(service.description, node.description, function.description))
             
        # Adds a constraint that forces one of the replication variables to take a value of one.
        for service in self.services:
            for function in service.get_vnfs(self.vnfs):
                dummy_installation = self.model.getVarByName(service.description + "_" + dummy_node.description + "_" + function.description + "_installation")
                vars_used = [self.model.getVarByName(service.description+ "_" + function.description + "_" + str(i) + "_replication") for i in range(1, self.max_replicas + 1)]
                self.model.addConstr(gp.quicksum(vars_used) + dummy_installation == 1, name = "replication_{}_{}".format(service.description, function.description))

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
                vars_used, params, dummy_installations = [], [], []
                for function in service.get_vnfs(self.vnfs):
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
                constr1 = self.model.getConstrByName("assignment_{}_{}".format(node.description, function.description))
                constr2 = self.model.getConstrByName("installation_{}_{}_{}".format(service.description, node.description, function.description))
                if constr1 != None and constr2 != None:
                    # To avoid adding arbitrary links due to zero cost edges we set the cost to a small value
                    if -(constr1.getAttr("Pi") + self.min_flow_param / service.throughput * constr2.getAttr("Pi")) == 0:
                        e.cost = 1e-6
                    else:
                        e.cost = -(constr1.getAttr("Pi") + self.min_flow_param / service.throughput * constr2.getAttr("Pi"))
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
        ass = [c for c in constrs if "assignment" in c.getAttr("ConstrName")]
        inst = [c for c in constrs if "installation_{}".format(path.service.description) in c.getAttr("ConstrName")]
        constrs = bw + [pf] + ass + inst
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
            elif "pathflow" in constrs[i].getAttr("ConstrName"):
                # This is just 1 since we sum all the path flow variables.
                coefficients[i] = 1
            elif "assignment" in constrs[i].getAttr("ConstrName"):
                # This is 1 if the path has the function hosted on a particular node.
                tokens = constrs[i].getAttr("ConstrName").split("_")
                node, function = tokens[1], tokens[2]
                if function in path.service.vnfs and params["components assigned"][function] == node:
                    coefficients[i] = 1
                else:
                    coefficients[i] = 0
            elif "installation" in constrs[i].getAttr("ConstrName").split("_"):
                tokens = constrs[i].getAttr("ConstrName").split("_")
                service, node, function = tokens[1], tokens[2], tokens[3]
                s = [s for s in self.services if s.description == service]
                assert len(s) == 1, "Service names should be unique"
                s = s[0]
                if params["components assigned"][function] == node:
                    coefficients[i] = self.min_flow_param/s.throughput
                else:
                    coefficients[i] = 0 
            else:
                raise ValueError("No other constraints should contain the path variable")

        # Adds the variable and column.
        self.model.addVar(column = gp.Column(coefficients, constrs), name = path.description + "_flow")

    def add_cut_from_model(self, item, m):
        """
        Given a cover cut model, adds the equivalent constraint to the linear restricted master problem.
        """
        if isinstance(item, VNF):
            # Gets variables whose value are zero.
            vars = [v.varName for v in m.getVars() if v.x == 0]
            # Gets equivalent variables from LRMP.
            fractional_vars = [v for v in self.model.getVars() if v.varName in vars]
            # Adds constraint.
            self.model.addConstr(gp.quicksum(fractional_vars[i] for i in range(len(fractional_vars))) >= 1, name = item.description + "_cover{}".format(item.cuts_generated))
        else:
            # Gets variables whose value are zero.
            vars = [v.varName for v in m.getVars() if v.x == 1]
            # Gets equivalent variables from LRMP.
            fractional_vars = [v for v in self.model.getVars() if v.varName in vars]
            # Adds constraint.
            self.model.addConstr(gp.quicksum(fractional_vars[i] for i in range(len(fractional_vars))) <= len(fractional_vars) - 1, name = item.description + "_cover{}".format(item.cuts_generated))
     

    def compute_optimality_gap(self):
        """
        Computes the dual gap using the upper and lower bounds.
        """
        return self.upper_bound - self.lower_bound/self.lower_bound

    def optimise(self, max_iterations: int = 100, tolerance: float = 0.05):
        """
        Finds schedule that optimises probability of success using column generation
        """
        terminate = False

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

        while terminate == False and k < max_iterations:
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

            # # Solves the cover cut problem for each function and adds the constraints.
            # for function in self.vnfs:
            #     if function.throughput != None:
            #         logging.info(" SOLVING COVER CUT PROBLEM FOR {}\n".format(function.description)) if self.verbose > 0 else None
            #         cov = self.find_cover_cut(function)
            #         if cov.objVal < 1:
            #             logging.info(" New cut for VNF {} found so adding.\n".format(function.description)) if self.verbose > 0 else None
            #             function.cuts_generated += 1
            #             self.add_cut_from_model(function, cov)
            #         else:
            #             logging.info(" No cut found for VNF {}.\n".format(function.description)) if self.verbose > 0 else None
            
            # # Solves the cover cut problem for each node and adds the constraints.
            # for node in self.nodes:
            #     if not isinstance(node, Dummy):
            #         logging.info(" SOLVING COVER CUT PROBLEM FOR {}\n".format(node.description)) if self.verbose > 0 else None
            #         cov = self.find_cover_cut_node(node)
            #         if cov.objVal < 1:
            #             logging.info(" New cut for VNF {} found so adding.\n".format(node.description)) if self.verbose > 0 else None
            #             node.cuts_generated += 1
            #             self.add_cut_from_model(node, cov)
            #         else:
            #             logging.info(" No cut found for VNF {}.\n".format(function.description)) if self.verbose > 0 else None     
            # Updates and resolves RMP.
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
        for node in nodes:
            assignments = {}
            for function in self.vnfs:
                if function.throughput == None:
                    var = self.model.getVarByName(node.description + "_" + function.description + "_assignment")
                    if var != None:
                        if var.x == 1:
                            assignments[function.description] = 1
                else:
                    max_instances = min(8, node.cpu // function.cpu)
                    for i in range(1, max_instances + 1):
                        var = self.model.getVarByName(node.description + "_" + function.description + "_" + str(i) + "_assignment")
                        if var != None:
                            if var.x == 1:
                                assignments[function.description] = i
            node.assignments = assignments

        for service in self.services:
            for path in service.graph.paths:
                path.flow = self.model.getVarByName(name = path.description + "_flow").x
                
    def to_json(self) -> dict:
        """
        Returns a json dictionary describing the model.
        """
        to_return = {}
        to_return["status"] = self.status
        to_return["objective"] = self.model.objVal
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

    
