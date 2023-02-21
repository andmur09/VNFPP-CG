from service_class.vnf import VNF
from topology.link import Link
from topology.network import Network
from topology.location import *
from service_class.graph import service_graph

inf = 1000000000
eps = 1e-9

class Service(object):
    """
    Class representing a service
        \param description          String description of vnf
        \param vnfs           List of vnfs used in service
        \param latency              Float defining the latency required for the service
        \param throughput           Float defining the throughput required for the service
        \param availability         Float definint the required latency for the service.
        \param graph                Instance of graph class representing the service for a given topology.
        \param source               Instance of Location, representing start point of service.
        \param sink                 Instance of Location, representing end point of service.
    """
    def __init__(self, description: str, vnfs: list, throughput: float, latency: float, availability: float, source: Location, sink: Location):
        self.description = description
        self.vnfs = vnfs
        self.throughput = throughput
        self.latency = latency
        self.availability = availability
        self.source = source
        self.sink = sink
        self.graph = None
    
    def add_vnf(self, vnf: VNF):
        """
        Adds a vnf to the service.
        """
        self.vnfs.append(vnf)
        
    def print(self):
        """
        Prints the service in a readable format.
        TODO: To be updated.
        """
        print("\n")
        print("Description: ", self.description)
        print("Required Latency: ", self.required_latency)
        print("Required Throughput: ", self.required_throughput)
        for vnf in self.vnfs:
            print("vnf: ", vnf.description)
            print("\tRequired CPU: ", vnf.required_cpu)
            print("\tRequired CPU: ", vnf.required_ram)
        print("\n")
    
    def make_graph(self, topology: Network):
        """
        Makes a graph representing the service in the network
        """
        n_layers = len(self.vnfs) + 1
        layers = []
        nodes, edges = [], []
        # Makes copy of network into n_components + 1 layers
        for i in range(n_layers):
            layer = topology.copy(topology.description + "_l{}".format(i))
            layer.save_as_dot()
            for node in layer.locations:
                node.description = node.description + "_l{}".format(i)
            nodes += layer.locations
            edges += layer.links
            layers.append(layer)
        
        # Adds edges connecting the nodes on layer l to layer l+1. Traversing this edge represents assigning component l to the node on layer l.
        serv_g = service_graph(self.description + "_graph", nodes, edges, n_layers)
        nodes = [n for n in topology.locations if isinstance(n, Node) == True]

        for node in nodes:
            for l in range(n_layers - 1):
                from_ = serv_g.get_location_by_description(node.description + "_l{}".format(l))
                to_ = serv_g.get_location_by_description(node.description + "_l{}".format(l+1))
                serv_g.add_link(Link(from_, to_, cost = 0, assignment_link = True))

        self.graph = serv_g