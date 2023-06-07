import numpy as np
from math import log
from time import time
from service_class.graph import service_path
from service_class.graph import multi_layered_graph
from topology.network import Network
from topology.location import Dummy, Node, Switch
from topology.link import Link
import gurobipy as gp
from gurobipy import GRB
import logging
import json
from math import ceil
import copy

inf = 1000000000
eps = 1e-9
env = gp.Env(empty=True)
env.setParam("OutputFlag",0)
env.start()

class CompactModel(object):
    """
    Description:    Class representing the datacenter optimisation problem via column generation.
    Parameters:     datacenter - Instance of Datacenter class to be optimised.
                    services - list of Service class representing the services to be hosted in the datacenter.
                    verbose - int verbosity of output.
                    results - string filename to use to log output.
    """
    def __init__(self, network: Network, vnfs: list, services: list, verbose: int = 1, logfile: str ='log.txt', max_replicas: int = 3,
                node_availability: float = 0.9999, min_flow_param: int = 10, weights = [100, 1]) -> None:
        """
        Initialises the optimisation problem
        """
        if verbose not in [0, 1, 2]:
            raise ValueError("Invalid verbosity level. Use 0 for no log, 1 for info and 2 for info and debug.")
        self.verbose = verbose
        logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.DEBUG, filemode='w') if self.verbose > 0 else None

        self.network = network

        # # Adds the dummy node to the network.
        # self.network.add_dummy_node()

        self.services = services
        # Makes set of VNF's containing only those used and without duplicates
        self.vnfs = set()
        for service in self.services:
            for f in service.get_vnfs(vnfs):
                self.vnfs.add(f)
        self.vnfs = list(self.vnfs)
        self.lower_bound = eps
        self.upper_bound = inf
        self.status = None
        self.runtime = None
        self.model = None
        self.max_replicas = max_replicas
        self.node_availability = node_availability
        self.min_flow_param = min_flow_param
        self.nodes = [n for n in self.network.locations if isinstance(n, Node) and not isinstance(n, Dummy)]
        self.n_nodes_used = None
        self.total_penalties = None
        self.availability_penalties = None
        self.throughput_penalties = None
        self.n_requests = None
        #Gets a dictionary calculating the max number of replicas of a vnf that can be hosted on a node.
        self.replica_dict = {f.description: {n.description: min([n.cpu //f.cpu, n.ram // f.ram]) for n in self.nodes} for f in self.vnfs}
        self.weights = weights
        self.graph = self.make_graph()
    
    def make_graph(self):
        """
        Makes a K layered graph, representing the topology.
        """
        # Finds the service with the most VNFs. The number of layers in the graph is this value + 1.
        n_layers = max([len(s.vnfs) for s in self.services]) + 1
        nodes, edges = [], []
        # Makes copy of network into n_components + 1 layers
        for i in range(n_layers):
            layer = self.network.copy(self.network.description + "_l{}".format(i))
            for node in layer.locations:
                node.description = node.description + "_l{}".format(i)
            # Adds opposing edges since edges are bidirectional:
            for edge in layer.links:
                opposite_edge = [e for e in layer.links if e.source == edge.sink and e.sink == edge.source]
                if not opposite_edge:
                    layer.links.append(Link(source = edge.sink, sink = edge.source))
            nodes += layer.locations
            edges += layer.links


        # Adds edges connecting the nodes on layer l to layer l+1. Traversing this edge represents assigning component l to the node on layer l.
        mlg = multi_layered_graph(self.network.description + "_graph", nodes, edges, self.network, n_layers)
        nodes = [n for n in self.network.locations if isinstance(n, Node) == True]

        for node in nodes:
            for l in range(n_layers - 1):
                from_ = mlg.get_location_by_description(node.description + "_l{}".format(l))
                to_ = mlg.get_location_by_description(node.description + "_l{}".format(l+1))
                mlg.add_link(Link(from_, to_, assignment_link = True))

        return mlg

    def get_vnf_by_description(self, description: str):
        """
        Given a string name of a vnf, returns the vnf from self.vnfs
        """
        for f in self.vnfs:
            if f.description == description:
                return f
        return None

    def build_initial_model(self, heuristic = None):
        """
        Builds and solves the restricted master problem given the initial approximation points.
        """
        start = time()
        self.model = gp.Model(self.network.description, env=env)

        # Adds variables, for LP relaxation makes variables continuous.
        for node in self.nodes:
            for function in self.vnfs:
                # Adds variable that says a function is installed on a node.
                self.model.addVar(vtype=GRB.INTEGER, obj = node.cost, name= node.description + "_" + function.description +  "_assignment")
 
        for service in self.services:
            flow_edges = [e for e in self.graph.links if e.assignment_link == False]
            assignment_edges = [e for e in self.graph.links if e.assignment_link == True]
            # Adds variable representing flow through an edge on a layer for a service s.
            for edge in flow_edges:
                self.model.addVar(name = service.description + "_" + edge.get_description() + "_flow")
            for edge in assignment_edges:
                self.model.addVar(name = service.description + "_" + edge.get_description() + "_assignmentflow")            
            # Adds availability penalty variable.
            self.model.addVar(lb = 0, ub = 1, vtype = GRB.BINARY, name = service.description + "_availabilitypenalty")
            # Adds throughput penalty variable.
            self.model.addVar(lb = 0, ub = 1, vtype = GRB.CONTINUOUS, name = service.description + "_throughputpenalty")
            used = []
            for i in range(len(service.vnfs)):
                for node in self.nodes:
                    # Adds service installation variables.
                    self.model.addVar(lb = 0, ub = 1, vtype=GRB.INTEGER, name= service.description + "_" + node.description + "_vnf" + str(i) + "_installation")
                for j in range(1, self.max_replicas + 1):
                    # Adds service replication variables.
                    self.model.addVar(lb = 0, ub = 1, vtype=GRB.INTEGER, name= service.description + "_vnf" + str(i) + "_" + str(j) + "_replication")
        self.model.update()

        # Adds constraint that the sum of CPU and RAM of the functions does not exceed the capacity of the nodes.
        for node in self.nodes:
            vars_used, cpus, rams = [], [], []
            for function in self.vnfs:
                vars_used.append(self.model.getVarByName(node.description + "_" + function.description + "_assignment"))
                cpus.append(function.cpu)
                rams.append(function.ram)
            self.model.addConstr(gp.quicksum(cpus[i] * vars_used[i] for i in range(len(vars_used))) <= node.cpu, name = "cpu_{}".format(node.description))
            self.model.addConstr(gp.quicksum(rams[i] * vars_used[i] for i in range(len(vars_used))) <= node.ram, name = "ram_{}".format(node.description))

        # Adds flow routing constraints:
        for service in self.services:
            for node in self.graph.locations:
                tokens = node.description.split("_")
                node_name, layer = tokens[0], int(tokens[1][1:])
                # If its the source node we have 1 unit of outgoing flow.
                if node_name == service.source.description and layer == 0:
                    out_edges, in_edges = self.graph.outgoing_edge(node), self.graph.incoming_edge(node)
                    out_vars, in_vars = [], []
                    penalty_var = self.model.getVarByName(service.description + "_throughputpenalty")
                    for o in out_edges:
                        if o.assignment_link == True:
                            out_vars.append(self.model.getVarByName(service.description + "_" + o.get_description() + "_assignmentflow"))
                        else:
                            out_vars.append(self.model.getVarByName(service.description + "_" + o.get_description() + "_flow"))
                    for i in in_edges:
                        if i.assignment_link == True:
                            in_vars.append(self.model.getVarByName(service.description + "_" + i.get_description() + "_assignmentflow"))
                        else:
                            in_vars.append(self.model.getVarByName(service.description + "_" + i.get_description() + "_flow"))
                    self.model.addConstr(gp.quicksum(out_vars) - gp.quicksum(in_vars) == 1 - penalty_var, name = "sourceflow_{}_{}".format(service.description, node.description))

                # If its the sink node we have 1 unit of incoming flow:
                elif node_name == service.sink.description and layer == len(service.vnfs):
                    out_edges, in_edges = self.graph.outgoing_edge(node), self.graph.incoming_edge(node)
                    out_vars, in_vars = [], []
                    penalty_var = self.model.getVarByName(service.description + "_throughputpenalty")
                    for o in out_edges:
                        if o.assignment_link == True:
                            out_vars.append(self.model.getVarByName(service.description + "_" + o.get_description() + "_assignmentflow"))
                        else:
                            out_vars.append(self.model.getVarByName(service.description + "_" + o.get_description() + "_flow"))
                    for i in in_edges:
                        if i.assignment_link == True:
                            in_vars.append(self.model.getVarByName(service.description + "_" + i.get_description() + "_assignmentflow"))
                        else:
                            in_vars.append(self.model.getVarByName(service.description + "_" + i.get_description() + "_flow"))
                    self.model.addConstr(gp.quicksum(out_vars) - gp.quicksum(in_vars) == penalty_var - 1, name = "sinkflow_{}_{}".format(service.description, node.description))
                
                # For all other nodes, flow is conserved:
                else:
                    out_edges, in_edges = self.graph.outgoing_edge(node), self.graph.incoming_edge(node)
                    out_vars, in_vars = [], []
                    for o in out_edges:
                        if o.assignment_link == True:
                            out_vars.append(self.model.getVarByName(service.description + "_" + o.get_description() + "_assignmentflow"))
                        else:
                            out_vars.append(self.model.getVarByName(service.description + "_" + o.get_description() + "_flow"))
                    for i in in_edges:
                        if i.assignment_link == True:
                            in_vars.append(self.model.getVarByName(service.description + "_" + i.get_description() + "_assignmentflow"))
                        else:
                            in_vars.append(self.model.getVarByName(service.description + "_" + i.get_description() + "_flow"))
                    self.model.addConstr(gp.quicksum(out_vars) - gp.quicksum(in_vars) == 0, name = "flowconservation_{}_{}".format(service.description, node.description))
        

        # Adds constraint that says the sum of all flows through each edge must not exceed the bandwidth capacity.
        flow_edges = [e for e in self.graph.links if e.assignment_link == False]
        for link in self.network.links:
            vars_used, params_used = [], []
            for edge in flow_edges:
                if self.graph.get_edge_from_original_network(edge) == link:
                    for service in self.services:
                        vars_used.append(self.model.getVarByName(service.description + "_" + edge.get_description() + "_flow"))
                        params_used.append(service.throughput)
            self.model.addConstr(gp.quicksum(params_used[i] * vars_used[i] for i in range(len(vars_used))) <= link.bandwidth, name = "bandwidth_{}".format(link.get_description()))

        # Adds a constraint that says the sum of flows through any instance of a VNF must not exceed it's processing capacity.
        assignment_edges = [e for e in self.graph.links if e.assignment_link == True]
        for node in self.nodes:
            for function in self.vnfs:
                flow_vars_used, flow_params = [], []
                # Gets the assignment edge variables for all services using that vnf.
                for service in self.services:
                    for edge in assignment_edges:
                        tokens = edge.source.description.split("_")
                        node_used, layer = tokens[0], int(tokens[1][1:])
                        if layer < len(service.vnfs) and service.vnfs[layer] == function.description and node_used == node.description:
                            flow_vars_used.append(self.model.getVarByName(service.description + "_" + edge.get_description() + "_assignmentflow"))
                            flow_params.append(service.throughput)
                assignment_var = self.model.getVarByName(node.description + "_" + function.description + "_assignment")
                self.model.addConstr(gp.quicksum(flow_vars_used[i] * flow_params[i] for i in range(len(flow_vars_used))) <= assignment_var * function.throughput, name="throughput_{}_{}".format(node.description, function.description))

        # # Adds a constraint that forces the installation variable of a service to be 1 if at least one path using that node is used to route flow.
        for service in self.services:
            for node in self.nodes:
                for i in range(len(service.vnfs)):
                    # Gets the assignment edge flow variables which assume the VNF to be hosted on the node.
                    for edge in assignment_edges:
                        tokens = edge.source.description.split("_")
                        node_used, layer = tokens[0], int(tokens[1][1:])
                        if layer < len(service.vnfs) and layer == i and node_used == node.description:
                            flow_var = self.model.getVarByName(service.description + "_" + edge.get_description() + "_assignmentflow")
                            flow_param = self.min_flow_param
                    installation_var = self.model.getVarByName(service.description + "_" + node.description + "_vnf" + str(i) + "_installation")
                    self.model.addConstr(flow_var * flow_param >= installation_var, name="installation_{}_{}_vnf{}".format(service.description, node.description, i))
             
        # Adds a constraint that forces one of the replication variables to take a value of one.
        for service in self.services:
            for i in range(len(service.vnfs)):
                vars_used = [self.model.getVarByName(service.description+ "_vnf" + str(i) + "_" + str(j) + "_replication") for j in range(1, self.max_replicas + 1)]
                self.model.addConstr(gp.quicksum(vars_used) == 1, name = "replication_{}_vnf{}".format(service.description, i))

        # Adds a constraint that constrains the number of replicas to be equal to the number of different nodes hosting that function.
        for service in self.services:
            for i in range(len(service.vnfs)):
                #dummy_installation = self.model.getVarByName(service.description + "_" + dummy_node.description + "_" + function.description + "_installation")
                replication_vars_used = [self.model.getVarByName(service.description + "_vnf" + str(i) + "_" + str(j) + "_replication") for j in range(1, self.max_replicas + 1)]
                installation_var_used = [self.model.getVarByName(service.description + "_" + n.description + "_vnf" + str(i) + "_installation") for n in self.nodes if isinstance(n, Dummy) == False]
                params = [i for i in range(1, self.max_replicas + 1)]
                self.model.addConstr(gp.quicksum(params[j] * replication_vars_used[j] for j in range(len(replication_vars_used))) <= gp.quicksum(installation_var_used), name = "nreplicas_{}_vnf{}".format(service.description, i))

        # Adds availability constraints for each service.
        for service in self.services:
            if service.availability != None:
                rhs = log(service.availability)
                vars_used, params = [], [], 
                for i in range(len(service.vnfs)):
                    for j in range(1, self.max_replicas + 1):
                        vars_used.append(self.model.getVarByName(service.description+ "_vnf" + str(i) + "_" + str(j) + "_replication"))
                        params.append(log(1 - (1 - self.node_availability * function.availability) ** j))
                penalty = self.model.getVarByName(service.description + "_availabilitypenalty")
                self.model.addConstr(gp.quicksum(params[i] * vars_used[i] for i in range(len(vars_used))) + penalty >= rhs, name = "availability_{}".format(service.description))

        # # Updates and optimises the model.
        self.model.update()
        av_pens, av_params = [], []
        tp_pens, tp_params = [], []
        for service in self.services:
            av_pens.append(self.model.getVarByName(service.description + "_availabilitypenalty"))
            av_params.append(service.n_subscribers)
            tp_pens.append(self.model.getVarByName(service.description + "_throughputpenalty"))
            tp_params.append(service.n_subscribers)
        
        # Gets the cpu usage for the objective
        vars_used, params = [], []
        for node in self.nodes:
            for function in self.vnfs:
                vars_used.append(self.model.getVarByName(node.description + "_" + function.description +  "_assignment"))
                params.append(function.cpu)

        # Makes the objective SLA violation costs.
        self.model.setObjective(self.weights[0] * (gp.quicksum(av_pens[i] * av_params[i] for i in range(len(av_pens))) + 
                                2 * gp.quicksum(tp_pens[i] * tp_params[i] for i in range(len(tp_pens)))) + 
                                self.weights[1] * (gp.quicksum(vars_used[i] for i in range(len(vars_used)))))
        self.model.optimize()

        logging.info(" Initial model built, solving.") if self.verbose > 0 else None
        self.model.update()
        if self.verbose == 2:
            self.model.write("{}.lp".format(self.model.getAttr("ModelName")))
        self.model.optimize()
        logging.info( " Finished in time {}.".format(time() - start)) if self.verbose > 0 else None
        if self.model.status == GRB.OPTIMAL:
            self.upper_bound = self.model.objVal
            logging.info( " Optimisation terminated successfully.") if self.verbose > 0 else None
            logging.info(' Objective: {}'.format(self.model.objVal)) if self.verbose > 0 else None
            logging.info(' Vars:') if self.verbose > 0 else None
            for v in self.model.getVars():
                if v.x != 0:
                    logging.info(" Variable {}: ".format(v.varName) + str(v.x)) if self.verbose > 0 else None
        else:
            logging.error(" Optimisation Failed - consult .ilp file") if self.verbose > 0 else None
            self.model.computeIIS()
            if self.verbose == 2:
                self.model.write("{}.ilp".format(self.model.getAttr("ModelName")))
            raise ValueError("Optimisation failed")

    def optimise(self):
        """
        Finds the optimal VNF placement.
        """
        self.build_initial_model()
        
    def parse_solution(self):
        """"
        If the optimisation was successful, it parses the solution and updates the relevant objects with the information.
        """
        nodes_used = []
        for function in self.vnfs:
            function.assignments = {}
        # Updates the assignments dictionary for each VNF outlining where, and how many instances have been assigned.
        for function in self.vnfs:
            for node in self.nodes:
                var = self.model.getVarByName(node.description + "_" + function.description + "_assignment")
                if var.x != 0:
                    function.assignments[node.description] = var.x
                    if node.description not in nodes_used:
                        nodes_used.append(node.description)
        
        self.n_nodes_used = len(nodes_used)
        total_penalties = 0
        availability_penalties, throughput_penalties = 0, 0
        for service in self.services:
            sla_violations = {}
            tp = self.model.getVarByName(service.description + "_throughputpenalty").x
            av = self.model.getVarByName(service.description + "_availabilitypenalty").x
            total_penalties += tp * service.n_subscribers
            total_penalties += av * service.n_subscribers
            throughput_penalties += tp * service.n_subscribers
            availability_penalties += av * service.n_subscribers
            sla_violations["throughput"] = tp
            sla_violations["availability"] = av
            service.sla_violations = sla_violations
        self.total_penalties = total_penalties
        self.availability_penalties = availability_penalties
        self.throughput_penalties = throughput_penalties
                
    def to_json(self) -> dict:
        """
        Returns a json dictionary describing the model.
        """
        to_return = {}
        to_return["status"] = self.status
        to_return["objective"] = self.model.objVal
        to_return["lower bound"] = self.lower_bound
        to_return["upper bound"] = self.upper_bound
        to_return["number of nodes used"] = self.n_nodes_used
        to_return["number of service requests"] = sum([s.n_subscribers for s in self.services])
        to_return["total penalties"] = self.total_penalties
        to_return["availability penalties"] = self.availability_penalties
        to_return["throughput penalties"] = self.throughput_penalties
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

    
