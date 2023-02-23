from topology.network import Network
import itertools
import graphviz as gvz

class service_graph(Network):
    """
    This class is used to make a service graph. Service graph can be used for optimising datacenter.
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
        source, sink = tokens[0].split("_")[0], tokens[1].split("_")[0]
        for link in self.network.links:
            if link.get_description() == "(" + source + "," + sink + ")":
                return link
        return None
    
    def get_node_and_function_from_assignment_edge(self, link):
        """
        For a given assignment edge it returns the function and node combination considered by traversing that edge.
        """
        assert link.assignment_link == True
        token = link.get_description()[1:].split(",")[0].split("_")
        node, layer = token[0], token[1]
        node = self.network.get_location_by_description(node)
        function = self.service.vnfs[int(layer[1:])]
        return node, function
        

    def add_path(self, path):
        """
        Adds a path to the list of found paths.
        """
        self.paths.append(path)
    
    def save_as_dot(self, filename = None):
        """
        saves the datacenter topology as a DOT to filename.dot
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
    This class represents a path on the above graph class.
    """
    id_iter = itertools.count()
    def __init__(self, description, locations, links, network, service, n_layers: int):
        super().__init__(description, locations, links)
        self.description = description + "_path" + str(next(service_path.id_iter))
        self.network = network
        self.service = service
        self.n_layers = n_layers
    
    def get_params(self):
        """
        Gets a dictionary of parameters required for the RMP. These are: the number of times each edge has been traversed in the path and the nodes that each required VNF is considered assigned to.
        """
        to_return = {"components assigned": {}, "times traversed": {}}
        # Gets the number of times each edge from the network topology is traversed across the layers of the graph.
        for edge1 in [e for e in self.network.links if e.assignment_link == False]:
            n = 0
            for edge2 in self.links:
                if edge1.source.description in edge2.source.description and edge1.sink.description in edge2.sink.description:
                    n += 1
            to_return["times traversed"][edge1.get_description()] = n
        # Gets which components are assigned to which nodes.
        for edge in self.links:
            tokens1, tokens2 = edge.source.description.split("_"), edge.sink.description.split("_")
            if tokens1[0] == tokens2[0]:
                l = int(tokens1[-1][-1])
                to_return["components assigned"][self.service.vnfs[l].description] = tokens1[0]
        return to_return
            
    def check_if_same(self, other_path):
        """
        Used to check for duplicate paths.
        """
        if self.get_params == other_path.get_params:
            return True
        return False
