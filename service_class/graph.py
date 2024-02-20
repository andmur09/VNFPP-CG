from topology.network import Network
import itertools
import graphviz as gvz
from service_class.vnf import VNF
from topology.location import Node

class multi_layered_graph(Network):
    """
    This class is used to make a multi-layered graph. This is similar to service_graph but not specific to any service.
    -------------
    Params:
        network:     topology.network.Network
                        Network to use
        n_layers:   int
                        Number of layers in the multi-layered graph. 
    """
    def __init__(self, name, locations, links, network, n_layers: int):
        super().__init__(name, locations, links)
        self.network = network
        self.n_layers = n_layers

    def get_edge_from_original_network(self, link):
        """
        For a given link in the service graph, returns the equivalent link in the network topology.
        """
        tokens = link.get_description()[1:-1].split(",")
        source, sink = tokens[0].split("_")[0], tokens[1].split("_")[0][1:]
        for link in self.network.links:
            if link.get_description() == "(" + source + ", " + sink + ")":
                return link
            elif link.get_description() == "(" + sink + ", " + source + ")":
                return link
        return None
    
    def get_node_and_function_from_assignment_edge(self, service, link, vnfs):
        """
        For a given assignment edge it returns the function and node combination considered by traversing that edge.
        """
        assert link.assignment_link == True
        token = link.get_description()[1:].split(",")[0].split("_")
        node, layer = token[0], token[1]
        node = self.network.get_location_by_description(node)
        function = service.get_vnfs(vnfs)[int(layer[1:])]
        return node, function
    
    def save_as_dot(self, filename = None):
        """
        saves the topology as a DOT to filename.dot
        """
        if filename != None:
            if filename[-5:] != ".dot":
                filename = filename + ".dot"
        else:
            filename = self.description + ".dot"

        plot = gvz.Digraph()
        for location in self.locations:
            plot.node(name=str(location.id), label=location.description)
        
        for link in self.links:
            plot.edge(str(link.source.id), str(link.sink.id), label = str(link.cost))

        with open(filename, "w") as f:
            f.write(plot.source)

class service_graph(Network):
    """
    This class is used to make a service graph for the CGP.
    -------------
    Params:
        network:     topology.network.Network
                        network to use
        service:    service_class.service.Service
                        SFC to use
        n_layers:   int
                        number of layers in the multi-layered graph. 
        paths:      list[service_graph.graph.service_path]
                        list of paths enumerated so far.
    """
    def __init__(self, name, locations, links, network, service, n_layers: int):
        super().__init__(name, locations, links)
        self.network = network
        self.service = service
        self.n_layers = n_layers
        self.paths = []

    def get_edge_from_original_network(self, link):
        """
        For a given link in the service graph, returns the equivalent link in the network topology.
        """
        tokens = link.get_description()[1:-1].split(",")
        source, sink = tokens[0].split("_")[0], tokens[1].split("_")[0][1:]
        for link in self.network.links:
            if link.get_description() == "(" + source + ", " + sink + ")":
                return link
            elif link.get_description() == "(" + sink + ", " + source + ")":
                return link
        return None
    
    def get_node_and_function_from_assignment_edge(self, link, vnfs):
        """
        For a given assignment edge it returns the function and node combination considered by traversing that edge.
        """
        assert link.assignment_link == True
        token = link.get_description()[1:].split(",")[0].split("_")
        node, layer = token[0], token[1]
        node = self.network.get_location_by_description(node)
        function = self.service.get_vnfs(vnfs)[int(layer[1:])]
        return node, function
        

    def add_path(self, path):
        """
        Adds a path to the list of found paths.
        """
        self.paths.append(path)
    
    def save_as_dot(self, filename = None):
        """
        saves the topology as a DOT to filename.dot
        """
        if filename != None:
            if filename[-5:] != ".dot":
                filename = filename + ".dot"
        else:
            filename = self.description + ".dot"

        plot = gvz.Digraph()
        for location in self.locations:
            plot.node(name=str(location.id), label=location.description)
        
        for link in self.links:
            plot.edge(str(link.source.id), str(link.sink.id), label = str(link.cost))

        with open(filename, "w") as f:
            f.write(plot.source)

class service_path(Network):
    """
    Class used to store paths on the service_graph.
    -------------
    Params:
        description:        str
                                string describing the path.
        network:            topology.network.Network
                                network to use
        service:            service_class.service.Service
                                SFC to use
        n_layers:           int
                                number of layers in the multi-layered graph. 
        latency_violated:   bool
                                True if the path violates the SFC latency constraint, else False.
        flow:               float
                                fraction of SFC flow routed down path.
    """
    id_iter = itertools.count()
    def __init__(self, description, locations, links, network, service, n_layers: int, latency_violated: bool = False):
        super().__init__(description, locations, links)
        self.description = description + "_path" + str(next(service_path.id_iter))
        self.network = network
        self.service = service
        self.n_layers = n_layers
        self.latency_violated = latency_violated
        self.flow = None
    
    def get_edge_from_original_network(self, link):
        """
        For a given link in the service graph, returns the equivalent link in the network topology.
        """
        tokens = link.get_description()[1:-1].split(",")
        source, sink = tokens[0].split("_")[0], tokens[1].split("_")[0][1:]
        for link in self.network.links:
            if link.get_description() == "(" + source + ", " + sink + ")":
                return link
            elif link.get_description() == "(" + sink + ", " + source + ")":
                return link
        return None

    def __str__(self):
        """
        Prints the path in a readable format.
        """
        # Gets the start and end node.
        start = None
        end = None

        for location in self.locations:
            outgoing_edges = [l for l in self.links if l.source == location]
            incoming_edges = [l for l  in self.links if l.sink == location]
            if not incoming_edges:
                start = location
            if not outgoing_edges:
                end = location

        outgoing_from_source = [l for l in self.links if l.source == start]
        assert len(outgoing_from_source) == 1, "Path can only have one outgoing edge from each node"
        path = [outgoing_from_source[0]]
        # Builds path
        current_node = path[-1].sink
        while current_node != end:
            outgoing_from_curr = [l for l in self.links if l.source == current_node]
            if len(outgoing_from_curr) != 1:
                self.save_as_dot()
            assert len(outgoing_from_curr) == 1, "Path can only have one outgoing edge from each node"
            path.append(outgoing_from_curr[0])
            current_node = path[-1].sink
        
        str_path = [e.get_description() for e in path]
        return " > ".join(str_path)
        
    def get_params(self):
        """
        Gets a dictionary of parameters required for the RMP. These are: the number of times each edge has been traversed in the path and the nodes that each required VNF is considered assigned to.
        """
        to_return = {"components assigned": {}, "times traversed": {}, "latency violated": self.latency_violated}
        # Gets the number of times each edge from the network topology is traversed across the layers of the graph.
        normal_edges = [e for e in self.links if e.assignment_link == False]
        assignment_edges = [e for e in self.links if e.assignment_link == True]
        for edge in normal_edges:
            oe = self.get_edge_from_original_network(edge)
            if oe != None:
                if oe.get_description() not in to_return["times traversed"].keys():
                    to_return["times traversed"][oe.get_description()] = 1
                else:
                    to_return["times traversed"][oe.get_description()] += 1
        # Gets which components are assigned to which nodes.
        for edge in assignment_edges:
            tokens1, tokens2 = edge.source.description.split("_"), edge.sink.description.split("_")
            if tokens1[0] == tokens2[0]:
                l = int(tokens1[-1][-1])
                to_return["components assigned"][l] = tokens1[0]
        return to_return

    def get_actual_path(self):
        """"
        Gets the list of edges used from the network in chronological order in the path.
        """
        # Gets the start and end node.
        start = None
        end = None
        for location in self.locations:
            outgoing_edges = [l for l in self.links if l.source == location]
            incoming_edges = [l for l  in self.links if l.sink == location]
            if not incoming_edges:
                start = location
            if not outgoing_edges:
                end = location

        outgoing_from_source = [l for l in self.links if l.source == start]
        assert len(outgoing_from_source) == 1, "Path can only have one outgoing edge from each node"
        path = [outgoing_from_source[0]]
        # Builds path
        current_node = path[-1].sink
        while current_node != end:
            outgoing_from_curr = [l for l in self.links if l.source == current_node]
            assert len(outgoing_from_curr) == 1, "Path can only have one outgoing edge from each node"
            path.append(outgoing_from_curr[0])
            current_node = path[-1].sink

        tidied_path = []
        for edge in path:
            source = edge.source.description.split("_")
            sink = edge.sink.description.split("_")
            # Removes assignment edges as they are not physical links.
            if source[0] != sink[0]:
                # Gets the actual link.
                source, sink = source[0], sink[0]
                tidied_path.append("({}, {})".format(source, sink))
        return tidied_path

    def to_json(self):
        """
        Returns the path as a json compatible dictionary.
        """
        return {"description": self.description, "links used": self.get_actual_path(), "flow": self.flow, "latency violated": self.latency_violated}

    def check_if_same(self, other_path):
        """
        Used to check for duplicate paths.
        """
        if self.__str__() == other_path.__str__():
            return True
        return False
    
    def check_if_using_assignment(self, vnf, node):
        """
        Returns True if path has VNF installed on a node, else False:
        """
        assignments = self.get_params()["components assigned"]
        # If the string description is provided, uses that.
        if isinstance(vnf, str) and isinstance(node, str):
            vnf_indexes = [i for i in range(len(self.service.vnfs)) if vnf == self.service.vnfs[i]]
            for i in vnf_indexes:
                if assignments[i] == node:
                    return True
        # If the objects are provided, uses that.
        elif isinstance(vnf, VNF) and isinstance(node, Node):
            vnf_indexes = [i for i in range(len(self.service.vnfs)) if vnf.description == self.service.vnfs[i]]
            for i in vnf_indexes:
                if assignments[i] == node.description:
                    return True
        else:
            raise ValueError("Invalid input type.")
        return False
    
    def count_times_using_assignment(self, vnf, node):
        """
        Returns True if path has VNF installed on a node, else False:
        """
        assignments = self.get_params()["components assigned"]
        # If the string description is provided, uses that.
        if isinstance(vnf, str) and isinstance(node, str):
            count = 0
            vnf_indexes = [i for i in range(len(self.service.vnfs)) if vnf == self.service.vnfs[i]]
            for i in vnf_indexes:
                try:
                    if assignments[i] == node:
                        # print("\nChecking VNF " + vnf + " and node " + node)
                        # print("Service VNFS", self.service.vnfs)
                        # print("VNF Indexes found", vnf_indexes)
                        # print("Assignments", assignments)
                        # print("Index", i + 1)
                        count += 1
                except KeyError:
                        print("\nDescription", self.description)
                        print("Checking VNF " + vnf + " and node " + node)
                        print("Service VNFS", self.service.vnfs)
                        print("VNF Indexes found", vnf_indexes)
                        print("Assignments", assignments)
                        print("Index", i + 1)
                        print("Assignment edges: ")
                        for edge in self.links:
                            if edge.assignment_link == True:
                                print(edge.get_description())
        # If the objects are provided, uses that.
        elif isinstance(vnf, VNF) and isinstance(node, Node):
            count = 0
            vnf_indexes = [i for i in range(len(self.service.vnfs)) if vnf.description == self.service.vnfs[i]]
            for i in vnf_indexes:
                try:
                    # print("\nChecking VNF " + vnf.description + " and node " + node.description)
                    # print("Service VNFS", self.service.vnfs)
                    # print("VNF Indexes found", vnf_indexes)
                    # print("Assignments", assignments)
                    # print("Index", i + 1)
                    if assignments[i] == node.description:
                        count += 1
                except KeyError:
                    print("\nDescription", self.description)
                    print("Checking VNF " + vnf.description + " and node " + node.description)
                    print("Service VNFS", self.service.vnfs)
                    print("VNF Indexes found", vnf_indexes)
                    print("Assignments", assignments)
                    print("Index", i + 1)
                    print("Assignment edges: ")
                    for edge in self.links:
                        if edge.assignment_link == True:
                            print(edge.get_description())
        return count