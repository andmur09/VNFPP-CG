import numpy as np
from math import log, exp
from scipy import stats
from time import time
from service_class.service import Service
from service_class.graph import service_path
from topology.datacenter import Datacenter
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
    def __init__(self, datacenter: Datacenter, services: list, verbose: int = 1, logfile: str ='log.txt') -> None:
        """
        Initialises the optimisation problem
        """
        if verbose not in [0, 1, 2]:
            raise ValueError("Invalid verbosity level. Use 0 for no log, 1 for info and 2 for info and debug.")
        self.verbose = verbose
        logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.DEBUG, filemode='w') if self.verbose == True else None

        self.datacenter = datacenter
        self.services = services
        self.lower_bound = eps
        self.upper_bound = inf
        self.status = None
        self.model = None
        
        # Initialises inner approximation for each service
        for service in services:
            service.add_graph(self.datacenter)

    def get_initial_paths(self):
        """
        Generates initial paths with arbitrarily high cost to kickstart algorithm.
        """
        for s in self.services:
            graph = s.graphs[self.datacenter.description]
            for link in graph.links:
                if "Dummy" in link.description:
                    # Sets link cost to dummy node to be arbitrarily small so that the path is always selected.
                    # This is a modelling convenience to ensure that there is a valid path.
                    link.cost = eps
            # Solves shortest path column generation problem
            m = self.column_generation(s)
            # Adds path to service graph
            self.add_path_from_model(s, m)
            for link in graph.links:
                if "Dummy" in link.description:
                    # From henceforth makes the cost arbitrarily high so that it is not selected.
                    link.cost = inf

    def add_path_from_model(self, service, model):
        """
        Given a service and a column generation model it adds the current path to the service graph.
        """
        graph = service.graphs[self.datacenter.description]
        links = [v for v in model.getVars() if "links" in v.varName]
        # From solution gets set of links
        links_i = [i for i in range(len(graph.links)) if links[i].x == 1]
        used_links = [graph.links[i] for i in range(len(graph.links)) if i in links_i]

        # From links gets set of nodes and adds path
        used_nodes = set()
        for link in used_links:
            if link.source not in used_nodes:
                used_nodes.add(link.source)
            if link.sink not in used_nodes:
                used_nodes.add(link.sink)
        used_nodes = list(used_nodes)

        # Counts each time an edge in the original topology has been traversed by the path
        times_traversed = {}
        for link1 in self.datacenter.links:
            z = 0
            for link2 in used_links:
                if link1.source.description == link2.source.description and link1.sink.description == link2.sink.description:
                    z += 1   
                elif link1.sink.description == link2.source.description and link1.source.description == link2.sink.description:
                    z += 1
            times_traversed[link1.description] = z
        
        # Makes binary vector representing assignment of component to node for the path:
        component_assignment = {}
        for component in service.components:
            component_assignment[str(component.description)] = {}
            for node in self.datacenter.get_locations_by_type("Node"):
                if node.description + "[" + component.description + "]" in [i.description for i in used_nodes]:
                    component_assignment[str(component.description)][str(node.description)] = 1
                else:
                    component_assignment[str(component.description)][str(node.description)] = 0
        path = service_path(self.datacenter.description + "_" + service.description, used_nodes, used_links, times_traversed, component_assignment)
        graph.add_path(path)
        if self.model != None:
            self.add_column_from_path(service, path)
    
    def add_column_from_path(self, service, path):
        """
        Takes a service graph path object and adds the equivalent column to the master problem.
        """
        c_tp = [self.model.getConstrByName("throughput_{}".format(service.description))]
        c_bw = [c for c in self.model.getConstrs() if "bandwidth" in c.getAttr("ConstrName")]
        c_assignment = [c for c in self.model.getConstrs() if "assignmentflow_{}".format(service.description) in c.getAttr("ConstrName")]

        coeff_tp = [1]
        # Gets the coefficients for the bandwidth constraints (equivalent to the number of times the path has traversed a particular link).
        coeffs_bw = []
        for c in c_bw:
            edge = c.getAttr("ConstrName")
            coeffs_bw.append(path.times_traversed[str(c.getAttr("ConstrName").split("_")[-1])])

        coeffs_assignment = []
        for c in c_assignment:
            tokens = c.getAttr("ConstrName").split("_")
            coeffs_assignment.append(path.component_assignment[tokens[2]][tokens[3]])

        constraints = c_tp + c_bw + c_assignment
        coefficients = coeff_tp + coeffs_bw + coeffs_assignment
        paths = service.graphs[self.datacenter.description].paths
        self.model.addVar(column = gp.Column(coefficients, constraints), name = service.description + "_flows_{}".format(len(paths)))


    def column_generation(self, service):
        """
        Solves the column generation problem for a service and finds the best new path to add to the master problem.
        """
        if self.model != None:
            pi = self.model.getConstrByName("throughput_{}".format(service.description)).getAttr("Pi")
        else:
            pi = 0
        m = gp.Model(self.datacenter.description + "_" + service.description, env=env)
        graph = service.graphs[self.datacenter.description]

        # Adds variables representing whether an edge has been used in a path or not:
        links = m.addMVar(shape=len(graph.links), vtype=GRB.BINARY, name="links")
        weights = np.array([l.cost for l in graph.links])

        # Gets indexes of links corresponding to ones leaving the source
        start = graph.get_start_node()
        outgoing = graph.outgoing_edge(start)
        start_indexes = [i for i in range(len(graph.links)) if graph.links[i] in outgoing]

        # Adds constraint that exactly one link leaving the source must be active
        m.addConstr(gp.quicksum([links[i] for i in start_indexes]) == 1)

        # Gets indexes of links corresponding to ones entering the sink
        end = graph.get_end_node()
        incoming = graph.incoming_edge(end)
        end_indexes = [i for i in range(len(graph.links)) if graph.links[i] in incoming]

        # Adds constraint that exactly one link entering the sink must be active
        m.addConstr(gp.quicksum([links[i] for i in end_indexes]) == 1)

        # Adds constraint that the sum of the flow into and out of every other edge must be conserved
        source_and_sink = [graph.locations.index(i) for i in [start, end]]
        for i in range(len(graph.locations)):
            if i not in source_and_sink:
                incoming = graph.incoming_edge(graph.locations[i])
                incoming_i = [i for i in range(len(graph.links)) if graph.links[i] in incoming]
                outgoing = graph.outgoing_edge(graph.locations[i])
                outgoing_i = [i for i in range(len(graph.links)) if graph.links[i] in outgoing]
                m.addConstr(gp.quicksum([links[i] for i in incoming_i]) == gp.quicksum([links[o] for o in outgoing_i]))
        
        m.setObjective(-pi + weights @ links, GRB.MINIMIZE)
        m.update()
        m.optimize()

        if m.status == GRB.OPTIMAL:
            logging.info(" Optimisation terminated successfully.") if self.verbose > 0 else None
            logging.info(' Ojective: {}'.format(m.objVal)) if self.verbose > 0 else None
            logging.info(' Vars:')
            vars = m.getVars()
            for i in range(len(vars)):
                if vars[i].x != 0:
                    logging.info("Description: {}, Value: {}".format(graph.links[i].description, vars[i].x)) if self.verbose > 0 else None
            return m
        else:
            logging.error(" Optimisation Failed - consult .ilp") if self.verbose > 0 else None
            m.computeIIS()
            m.write("{}.ilp".format(m.getAttr("ModelName")))
            raise ValueError("Optimisation failed")

    def update_duals(self):
        """
        Updates the edges on the service graphs in line with current dual solution from the master problem.
        """
        for s in self.services:
            graph = s.graphs[self.datacenter.description]
            for link in graph.links:
                if "Component" not in link.description:
                    if link.source.description != link.sink.description:
                        try:
                            link.cost = -self.model.getConstrByName("bandwidth_{}".format(link.description)).getAttr("Pi")
                        except:
                            opposing = "({}, {})".format(link.sink.description, link.source.description)
                            link.cost = -self.model.getConstrByName("bandwidth_{}".format(opposing)).getAttr("Pi")
                elif "Component" in link.description:
                    # Gets the description of the component assigned
                    component = [n.get_component_assigned() for n in [link.source, link.sink] if isinstance(n, Node) == True]
                    component = [c for c in component if c != None]
                    assert len(component) == 1
                    component = component[0]
                    # Gets the description of the node that the component is assigned on
                    node = [n.description for n in [link.source, link.sink] if "Component" not in n.description]
                    assert len(node) == 1
                    node = node[0]
                    # Gets the constraint using the service, component and node description
                    constraint = self.model.getConstrByName("assignmentflow_{}_{}_{}".format(s.description, component, node))
                    link.cost = -constraint.getAttr("Pi")/2
            graph.save_as_dot()

    def build_initial_model(self):
        """
        Builds and solves the restricted master problem given the initial approximation points.
        """
        self.model = gp.Model(self.datacenter.description, env=env)
        nodes = self.datacenter.get_locations_by_type("Node")
        n_nodes = len(nodes)
        rep_constants = [10 if "Dummy" in n.description else 1 for n in nodes]

        components = set()
        # Adds flow vector for each service where each element represents a path associated with that service
        # Also makes set of components for all services so that no duplicate components are considered
        for service in self.services:
            graph = service.graphs[self.datacenter.description]
            self.model.addVar(vtype=GRB.CONTINUOUS, name= service.description + "_flows_1")
            for component in service.components:
                components.add(component)
        components = list(components)

        # For LP relaxation makes variables continuous
        for component in components:
            self.model.addMVar(shape=n_nodes, lb = 0, ub = 1, vtype=GRB.CONTINUOUS, name=component.description + "_assignment")
        self.model.update()
        
        # # Adds constraint that the sum of flows for each service must be greater than the required throughput for the service
        for service in self.services:
            flows = [v for v in self.model.getVars() if service.description + "_flows" in v.varName]
            self.model.addConstr(gp.quicksum(flows) >= service.required_throughput, name="throughput_{}".format(service.description))

        # Adds a constraint that says that the sum of all flows through the edge must be less than the bandwidth:
        for i in range(len(self.datacenter.links)):
            path_vars = []
            coefficients = []
            for service in self.services:
                path_vars.append([v for v in self.model.getVars() if service.description + "_flows" in v.varName])
                graph = service.graphs[self.datacenter.description]
                coefficients.append([path.times_traversed[self.datacenter.links[i].description] for path in graph.paths])
            self.model.addConstr(gp.quicksum(coefficients[s][p]*path_vars[s][p] for s in range(len(self.services)) for p in range(len(path_vars[s]))) <= self.datacenter.links[i].bandwidth, name="bandwidth_{}".format(self.datacenter.links[i].description))
        
        # Adds constraint that forces flows to be equal to zero for any path not containing a node that a required component is assigned to
        for n in range(len(nodes)):
            for s in self.services:
                for c in s.components:
                    #print([c.description for c in s.components])
                    y = self.model.getVarByName(c.description + "_assignment"+"[{}]".format(n))
                    x = [v for v in self.model.getVars() if s.description + "_flows" in v.varName]
                    graph = s.graphs[self.datacenter.description]
                    assignments = [p.component_assignment for p in graph.paths]
                    #print(assignments)
                    alpha = [assignments[g][c.description][nodes[n].description] for g in range(len(x))]
                    self.model.addConstr(gp.quicksum(alpha[i]*x[i] for i in range(len(x))) <= s.required_throughput * y, name="assignmentflow_{}_{}_{}".format(s.description, c.description, nodes[n].description))

        # Adds a constraint that says that a component must be assigned to x different nodes where x is the replica count
        # To enable solutions even when it is not possible to assign enough replicas, we multiply the variable for the assignment of a component
        # to a dummy node by an arbitrarily high constant.
        for component in components:
            assignment_vars = [v for v in self.model.getVars() if component.description + "_assignment" in v.varName]
            self.model.addConstr(gp.quicksum(assignment_vars[i] * rep_constants[i] for i in range(len(assignment_vars))) >= component.replica_count, name = "replicas_{}".format(component.description))

        # Adds a constraint that says that the sum of component requirements running on a node must not exceed the capacity.
        for i in range(len(nodes)):
            assignment_variables = [self.model.getVarByName(component.description + "_assignment"+"[{}]".format(i)) for component in components]
            # For CPU
            requirements = [component.requirements["cpu"] for component in components]
            self.model.addConstr(gp.quicksum([assignment_variables[i]*requirements[i] for i in range(len(components))]) <= nodes[i].cpu, name="capacity_{}_{}".format("cpu", nodes[i].description))
            # For RAM
            requirements = [component.requirements["ram"] for component in components]
            self.model.addConstr(gp.quicksum([assignment_variables[i]*requirements[i] for i in range(len(components))]) <= nodes[i].ram, name="capacity_{}_{}".format("ram", nodes[i].description))
            # Adds a constraint that fixes all assignment variables to zero whenever a node is not active (used to simulate node failure)
            if nodes[i].active == False:
                for v in assignment_variables:
                    self.model.addConstr(v == 0)

        #Sets objective to minimise node rental costs
        node_rental_costs = []
        node_assignments = []
        for i in range(len(nodes)):
            node_rental_costs.append(nodes[i].cost)
            node_assignments.append([self.model.getVarByName(component.description + "_assignment"+"[{}]".format(i)) for component in components])
        self.model.setObjective(gp.quicksum(node_rental_costs[i] * node_assignments[i][j] for i in range(len(nodes)) for j in range(len(components))), GRB.MINIMIZE)
        
        self.model.update()
        self.model.optimize()
            
        logging.info(" Initial model built, solving.") if self.verbose > 0 else None
        self.model.update()
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            logging.info( " Optimisation terminated successfully.") if self.verbose > 0 else None
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
        
        # Uses heuristics to generate intitial points.
        logging.info(" Attempting to generate initial point.") if self.verbose > 0 else None
        self.get_initial_paths()
        logging.info(" Initialisation completed.\n")
        for service in self.services:
            graph = service.graphs[self.datacenter.description]
            for path in graph.paths:
                path.save_as_dot("path")

        # Solves restricted master problem using initial points and saves solution.
        logging.info(" BUILDING INITIAL MODEL.\n") if self.verbose > 0 else None
        self.build_initial_model()
        logging.info(" Updating service graphs with dual information.\n")
        self.update_duals()

        no_iterations = 1
        self.upper_bound = self.model.objVal
        
        lb = self.upper_bound
        statuses = []
        # Solves the column generation problem for each sub problem.
        for service in self.services:
            logging.info(" SOLVING COLUMN GENERATION FOR {}\n".format(service.description)) if self.verbose > 0 else None
            try:
                cg = self.column_generation(service)
                if cg.objVal < 0:
                    logging.info(" New path for service {} has negative reduced cost so adding.\n".format(service.description)) if self.verbose > 0 else None
                    self.add_path_from_model(service, cg)
                lb += cg.objVal
                statuses.append(True)
            except ValueError:
                lb += -inf
                statuses.append(False)
        # If lower bound is better than current lower bound it updates.
        if lb >= self.lower_bound and all(statuses) == True:
            self.lower_bound = lb
        bound = self.compute_optimality_gap()

        # If all of the sub problems resulted in non-negative reduced cost we can terminate.
        # We define an alowable tolerance on the reduced cost which we check against.
        while (bound > tolerance or bound < 0) and no_iterations < max_iterations:
            no_iterations += 1
            # If not satisfied we can run the master problem with the new columns added
            self.model.update()
            self.model.write("{}.lp".format(self.model.getAttr("ModelName")))
            logging.info(" SOLVING RMP ON ITERATION {}\n".format(no_iterations)) if self.verbose > 0 else None
            self.model.optimize()
            if self.model.status == GRB.OPTIMAL:
                logging.info( " Optimisation terminated successfully.") if self.verbose > 0 else None
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

            logging.info(" Updating service graphs with dual information.\n")
            self.update_duals()

            self.upper_bound = self.model.objVal
            logging.info(" UPDATING UPPER BOUND: {}".format(self.model.objVal)) if self.verbose > 0 else None

            lb = self.upper_bound
            statuses = []
            # Solves the column generation problem for each sub problem.
            for service in self.services:
                logging.info(" SOLVING COLUMN GENERATION FOR {}\n".format(service.description)) if self.verbose > 0 else None
                try:
                    cg = self.column_generation(service)
                    if cg.objVal < 0:
                        logging.info(" New path for service {} has negative reduced cost so adding.\n".format(service.description)) if self.verbose > 0 else None
                        self.add_path_from_model(service, cg)
                    lb += cg.objVal
                    statuses.append(True)
                except ValueError:
                    lb += -inf
                    statuses.append(False)
            # If lower bound is better than current lower bound it updates.
            if lb >= self.lower_bound and all(s tatuses) == True:
                self.lower_bound = lb
            bound = self.compute_optimality_gap()

        if (bound <= tolerance and bound >= 0) and self.model.status == GRB.OPTIMAL:
            logging.info(" Final Optimisation terminated sucessfully") if self.verbose > 0 else None
            logging.info(' Objective: {}'.format(self.model.objVal)) if self.verbose > 0 else None
            logging.info(" Probability: {}".format(exp(-self.model.objVal))) if self.verbose > 0 else None
            logging.info(' Vars:')
            for v in self.model.getVars():
                if "_lam_" in v.varName and v.x == 0:
                    continue
                else:
                    logging.info(" Variable {}: ".format(v.varName) + str(v.x)) if self.verbose > 0 else None
            self.status = "Optimal"
        else:
            logging.warning(" Failed to satisfy bound on optimality within required iterations. Try increasing allowable iterations.") if self.verbose > 0 else None
            logging.info(' Objective: {}'.format(self.model.objVal)) if self.verbose > 0 else None
            logging.info(" Probability: {}".format(exp(-self.model.objVal))) if self.verbose > 0 else None
            logging.info(' Vars:') if self.verbose > 0 else None
            for v in self.model.getVars():
                if "_lam_" in v.varName and v.x == 0:
                    continue
                else:
                    logging.info(" Variable {}: ".format(v.varName) + str(v.x)) if self.verbose > 0 else None
            self.status = "Failed"
