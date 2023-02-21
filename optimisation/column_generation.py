import numpy as np
from math import log, exp
from scipy import stats
from time import time
from service_class.service import Service
from service_class.graph import service_graph, service_path
from topology.network import Network
from topology.location import Node
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
    def __init__(self, network: Network, services: list, verbose: int = 1, logfile: str ='log.txt') -> None:
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
        
        # Initialises inner approximation for each service
        for service in services:
            service.make_graph(self.network)

    def pricing_problem(self, service):
        """
        Solves the pricing problem for a service and finds the best new path to add to the master problem.
        """
        graph = service.graph
        graph.save_as_dot()
        m = gp.Model(graph.description, env=env)
        w = np.array([l.cost for l in graph.links])
        x = m.addMVar(shape = len(graph.links), vtype = GRB.BINARY, name="links")

        # Source flow constraint.
        # Gets the service source node.
        source = graph.get_location_by_description(service.source.description + "_l0")
        # Gets indexes of outgoing links.
        indexes = [graph.links.index(o) for o in graph.outgoing_edge(source)]
        # Adds constraint such that only one edge leaving the source must be active.
        m.addConstr(gp.quicksum(x[i] for i in indexes) == 1, name = "source")

        # Sink flow constraint.
        sink = graph.get_location_by_description(service.sink.description + "_l{}".format(graph.n_layers - 1))
        # Gets indexes of incoming links.
        indexes = [graph.links.index(i) for i in graph.incoming_edge(sink)]
        # Adds constraint such that only one edge into the sink must be active.
        m.addConstr(gp.quicksum(x[i] for i in indexes) == 1, name = "sink")

        # Flow conservation constraint.
        for node in graph.locations:
            if node not in [source, sink]:
                # Gets indexes of incoming and outgoing edges.
                o_indexes = [graph.links.index(o) for o in graph.outgoing_edge(node)]
                i_indexes = [graph.links.index(i) for i in graph.incoming_edge(node)]
                m.addConstr(gp.quicksum(x[o] for o in o_indexes) == gp.quicksum(x[i] for i in i_indexes), name = "conservation")
        
        # Adds latency constraint.
        lat = np.array([l.latency for l in graph.links])
        m.addConstr(lat @ x <= service.latency, name = "latency")

        # Gets dual variable associated with constraint that at least one path must be used.
        if self.model != None:
            pi = self.model.getConstrByName("throughput_{}".format(service.description)).getAttr("Pi")
        else:
            pi = 0

        # Sets objective to reduced cost.
        m.setObjective(-pi + w @ x, GRB.MINIMIZE)
        m.update()
        m.optimize()

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
        path = service_path(service.description, used_nodes, used_edges, self.network, n_layers = graph.n_layers)
        path.save_as_dot()
        graph.add_path(path)

    # def update_duals(self):
    #     """
    #     Updates the edges on the service graphs in line with current dual solution from the master problem.
    #     """
    #     for s in self.services:
    #         graph = s.graphs[self.network.description]
    #         for link in graph.links:
    #             if "Component" not in link.description:
    #                 if link.source.description != link.sink.description:
    #                     try:
    #                         link.cost = -self.model.getConstrByName("bandwidth_{}".format(link.description)).getAttr("Pi")
    #                     except:
    #                         opposing = "({}, {})".format(link.sink.description, link.source.description)
    #                         link.cost = -self.model.getConstrByName("bandwidth_{}".format(opposing)).getAttr("Pi")
    #             elif "Component" in link.description:
    #                 # Gets the description of the component assigned
    #                 component = [n.get_component_assigned() for n in [link.source, link.sink] if isinstance(n, Node) == True]
    #                 component = [c for c in component if c != None]
    #                 assert len(component) == 1
    #                 component = component[0]
    #                 # Gets the description of the node that the component is assigned on
    #                 node = [n.description for n in [link.source, link.sink] if "Component" not in n.description]
    #                 assert len(node) == 1
    #                 node = node[0]
    #                 # Gets the constraint using the service, component and node description
    #                 constraint = self.model.getConstrByName("assignmentflow_{}_{}_{}".format(s.description, component, node))
    #                 link.cost = -constraint.getAttr("Pi")/2
    #         graph.save_as_dot()

    # def build_initial_model(self):
    #     """
    #     Builds and solves the restricted master problem given the initial approximation points.
    #     """
    #     self.model = gp.Model(self.network.description, env=env)
    #     nodes = self.network.get_locations_by_type("Node")
    #     n_nodes = len(nodes)
    #     rep_constants = [10 if "Dummy" in n.description else 1 for n in nodes]

    #     components = set()
    #     # Adds flow vector for each service where each element represents a path associated with that service
    #     # Also makes set of components for all services so that no duplicate components are considered
    #     for service in self.services:
    #         graph = service.graphs[self.network.description]
    #         self.model.addVar(vtype=GRB.CONTINUOUS, name= service.description + "_flows_1")
    #         for component in service.components:
    #             components.add(component)
    #     components = list(components)

    #     # For LP relaxation makes variables continuous
    #     for component in components:
    #         self.model.addMVar(shape=n_nodes, lb = 0, ub = 1, vtype=GRB.CONTINUOUS, name=component.description + "_assignment")
    #     self.model.update()
        
    #     # # Adds constraint that the sum of flows for each service must be greater than the required throughput for the service
    #     for service in self.services:
    #         flows = [v for v in self.model.getVars() if service.description + "_flows" in v.varName]
    #         self.model.addConstr(gp.quicksum(flows) >= service.required_throughput, name="throughput_{}".format(service.description))

    #     # Adds a constraint that says that the sum of all flows through the edge must be less than the bandwidth:
    #     for i in range(len(self.network.links)):
    #         path_vars = []
    #         coefficients = []
    #         for service in self.services:
    #             path_vars.append([v for v in self.model.getVars() if service.description + "_flows" in v.varName])
    #             graph = service.graphs[self.network.description]
    #             coefficients.append([path.times_traversed[self.network.links[i].description] for path in graph.paths])
    #         self.model.addConstr(gp.quicksum(coefficients[s][p]*path_vars[s][p] for s in range(len(self.services)) for p in range(len(path_vars[s]))) <= self.network.links[i].bandwidth, name="bandwidth_{}".format(self.network.links[i].description))
        
    #     # Adds constraint that forces flows to be equal to zero for any path not containing a node that a required component is assigned to
    #     for n in range(len(nodes)):
    #         for s in self.services:
    #             for c in s.components:
    #                 #print([c.description for c in s.components])
    #                 y = self.model.getVarByName(c.description + "_assignment"+"[{}]".format(n))
    #                 x = [v for v in self.model.getVars() if s.description + "_flows" in v.varName]
    #                 graph = s.graphs[self.network.description]
    #                 assignments = [p.component_assignment for p in graph.paths]
    #                 #print(assignments)
    #                 alpha = [assignments[g][c.description][nodes[n].description] for g in range(len(x))]
    #                 self.model.addConstr(gp.quicksum(alpha[i]*x[i] for i in range(len(x))) <= s.required_throughput * y, name="assignmentflow_{}_{}_{}".format(s.description, c.description, nodes[n].description))

    #     # Adds a constraint that says that a component must be assigned to x different nodes where x is the replica count
    #     # To enable solutions even when it is not possible to assign enough replicas, we multiply the variable for the assignment of a component
    #     # to a dummy node by an arbitrarily high constant.
    #     for component in components:
    #         assignment_vars = [v for v in self.model.getVars() if component.description + "_assignment" in v.varName]
    #         self.model.addConstr(gp.quicksum(assignment_vars[i] * rep_constants[i] for i in range(len(assignment_vars))) >= component.replica_count, name = "replicas_{}".format(component.description))

    #     # Adds a constraint that says that the sum of component requirements running on a node must not exceed the capacity.
    #     for i in range(len(nodes)):
    #         assignment_variables = [self.model.getVarByName(component.description + "_assignment"+"[{}]".format(i)) for component in components]
    #         # For CPU
    #         requirements = [component.requirements["cpu"] for component in components]
    #         self.model.addConstr(gp.quicksum([assignment_variables[i]*requirements[i] for i in range(len(components))]) <= nodes[i].cpu, name="capacity_{}_{}".format("cpu", nodes[i].description))
    #         # For RAM
    #         requirements = [component.requirements["ram"] for component in components]
    #         self.model.addConstr(gp.quicksum([assignment_variables[i]*requirements[i] for i in range(len(components))]) <= nodes[i].ram, name="capacity_{}_{}".format("ram", nodes[i].description))
    #         # Adds a constraint that fixes all assignment variables to zero whenever a node is not active (used to simulate node failure)
    #         if nodes[i].active == False:
    #             for v in assignment_variables:
    #                 self.model.addConstr(v == 0)

    #     #Sets objective to minimise node rental costs
    #     node_rental_costs = []
    #     node_assignments = []
    #     for i in range(len(nodes)):
    #         node_rental_costs.append(nodes[i].cost)
    #         node_assignments.append([self.model.getVarByName(component.description + "_assignment"+"[{}]".format(i)) for component in components])
    #     self.model.setObjective(gp.quicksum(node_rental_costs[i] * node_assignments[i][j] for i in range(len(nodes)) for j in range(len(components))), GRB.MINIMIZE)
        
    #     self.model.update()
    #     self.model.optimize()
            
    #     logging.info(" Initial model built, solving.") if self.verbose > 0 else None
    #     self.model.update()
    #     self.model.optimize()
    #     if self.model.status == GRB.OPTIMAL:
    #         logging.info( " Optimisation terminated successfully.") if self.verbose > 0 else None
    #         logging.info(' Objective: {}'.format(self.model.objVal)) if self.verbose > 0 else None
    #         logging.info(' Vars:') if self.verbose > 0 else None
    #         for v in self.model.getVars():
    #             if v.x != 0:
    #                 logging.info(" Variable {}: ".format(v.varName) + str(v.x)) if self.verbose > 0 else None
    #     else:
    #         logging.error(" Optimisation Failed - consult .ilp file") if self.verbose > 0 else None
    #         self.model.computeIIS()
    #         self.model.write("{}.ilp".format(self.model.getAttr("ModelName")))
    #         raise ValueError("Optimisation failed")
   
    def compute_optimality_gap(self):
        logging.info(" Computing current optimality gap:") if self.verbose > 0 else None
        logging.info(" Lower bound: {}".format(self.lower_bound)) if self.verbose > 0 else None
        logging.info(" Upper bound: {}".format(self.upper_bound)) if self.verbose > 0 else None
        gap = (self.upper_bound - self.lower_bound)/self.lower_bound
        logging.info(" Gap: {}\n".format(gap)) if self.verbose > 0 else None
        return gap

    def optimise(self, max_iterations: int = 10, tolerance: float = 0.01):
        """
        Finds schedule that optimises probability of success using column generation
        """
        start = time()
        
        # Adding initial path.
        logging.info(" Attempting to generate initial path.") if self.verbose > 0 else None
        for service in self.services:
            cgp = self.pricing_problem(service)
            self.add_path_from_model(service, cgp)
        logging.info(" Initialisation completed.\n")

        # Solves restricted master problem using initial points and saves solution.
        logging.info(" BUILDING INITIAL MODEL.\n") if self.verbose > 0 else None
        self.build_initial_model()
        logging.info(" Updating service graphs with dual information.\n")
        self.update_duals()

        # no_iterations = 1
        # self.upper_bound = self.model.objVal
        
        # lb = self.upper_bound
        # statuses = []
        # # Solves the column generation problem for each sub problem.
        # for service in self.services:
        #     logging.info(" SOLVING COLUMN GENERATION FOR {}\n".format(service.description)) if self.verbose > 0 else None
        #     try:
        #         cg = self.pricing_problem(service)
        #         if cg.objVal < 0:
        #             logging.info(" New path for service {} has negative reduced cost so adding.\n".format(service.description)) if self.verbose > 0 else None
        #             self.add_path_from_model(service, cg)
        #         lb += cg.objVal
        #         statuses.append(True)
        #     except ValueError:
        #         lb += -inf
        #         statuses.append(False)
        # # If lower bound is better than current lower bound it updates.
        # if lb >= self.lower_bound and all(statuses) == True:
        #     self.lower_bound = lb
        # bound = self.compute_optimality_gap()

        # # If all of the sub problems resulted in non-negative reduced cost we can terminate.
        # # We define an alowable tolerance on the reduced cost which we check against.
        # while (bound > tolerance or bound < 0) and no_iterations < max_iterations:
        #     no_iterations += 1
        #     # If not satisfied we can run the master problem with the new columns added
        #     self.model.update()
        #     self.model.write("{}.lp".format(self.model.getAttr("ModelName")))
        #     logging.info(" SOLVING RMP ON ITERATION {}\n".format(no_iterations)) if self.verbose > 0 else None
        #     self.model.optimize()
        #     if self.model.status == GRB.OPTIMAL:
        #         logging.info( " Optimisation terminated successfully.") if self.verbose > 0 else None
        #         logging.info(' Objective: {}'.format(self.model.objVal)) if self.verbose > 0 else None
        #         logging.info(' Vars:') if self.verbose > 0 else None
        #         for v in self.model.getVars():
        #             if v.x != 0:
        #                 logging.info(" Variable {}: ".format(v.varName) + str(v.x)) if self.verbose > 0 else None
        #     else:
        #         logging.error(" Optimisation Failed - consult .ilp file") if self.verbose > 0 else None
        #         self.model.computeIIS()
        #         self.model.write("{}.ilp".format(self.model.getAttr("ModelName")))
        #         raise ValueError("Optimisation failed")

        #     logging.info(" Updating service graphs with dual information.\n")
        #     self.update_duals()

        #     self.upper_bound = self.model.objVal
        #     logging.info(" UPDATING UPPER BOUND: {}".format(self.model.objVal)) if self.verbose > 0 else None

        #     lb = self.upper_bound
        #     statuses = []
        #     # Solves the column generation problem for each sub problem.
        #     for service in self.services:
        #         logging.info(" SOLVING COLUMN GENERATION FOR {}\n".format(service.description)) if self.verbose > 0 else None
        #         try:
        #             cg = self.pricing_problem(service)
        #             if cg.objVal < 0:
        #                 logging.info(" New path for service {} has negative reduced cost so adding.\n".format(service.description)) if self.verbose > 0 else None
        #                 self.add_path_from_model(service, cg)
        #             lb += cg.objVal
        #             statuses.append(True)
        #         except ValueError:
        #             lb += -inf
        #             statuses.append(False)
        #     # If lower bound is better than current lower bound it updates.
        #     if lb >= self.lower_bound and all(s tatuses) == True:
        #         self.lower_bound = lb
        #     bound = self.compute_optimality_gap()

        # if (bound <= tolerance and bound >= 0) and self.model.status == GRB.OPTIMAL:
        #     logging.info(" Final Optimisation terminated sucessfully") if self.verbose > 0 else None
        #     logging.info(' Objective: {}'.format(self.model.objVal)) if self.verbose > 0 else None
        #     logging.info(" Probability: {}".format(exp(-self.model.objVal))) if self.verbose > 0 else None
        #     logging.info(' Vars:')
        #     for v in self.model.getVars():
        #         if "_lam_" in v.varName and v.x == 0:
        #             continue
        #         else:
        #             logging.info(" Variable {}: ".format(v.varName) + str(v.x)) if self.verbose > 0 else None
        #     self.status = "Optimal"
        # else:
        #     logging.warning(" Failed to satisfy bound on optimality within required iterations. Try increasing allowable iterations.") if self.verbose > 0 else None
        #     logging.info(' Objective: {}'.format(self.model.objVal)) if self.verbose > 0 else None
        #     logging.info(" Probability: {}".format(exp(-self.model.objVal))) if self.verbose > 0 else None
        #     logging.info(' Vars:') if self.verbose > 0 else None
        #     for v in self.model.getVars():
        #         if "_lam_" in v.varName and v.x == 0:
        #             continue
        #         else:
        #             logging.info(" Variable {}: ".format(v.varName) + str(v.x)) if self.verbose > 0 else None
        #     self.status = "Failed"
