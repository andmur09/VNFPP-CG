from lib2to3.pgen2.token import PERCENTEQUAL
from matplotlib.pyplot import vlines

from numpy import percentile
from service_class.vnf import VNF
from topology.link import Link
from topology.network import Network
from topology.location import *
from service_class.graph import service_graph
from service_class.graph import service_path
import json
inf = 1000000000
eps = 1e-9

class Service(object):
    """
    Class representing a service
        \param description          String description of vnf
        \param vnfs                 List of vnfs used in service
        \param latency              Float defining the latency required for the service
        \param throughput           Float defining the throughput required for the service
        \param percentage traffic   Float defining the percentage of traffic for this given service for a network load.
        \param availability         Float definint the required latency for the service.
        \param graph                Instance of graph class representing the service for a given topology.
        \param source               Instance of Location, representing start point of service.
        \param sink                 Instance of Location, representing end point of service.
    """
    def __init__(self, description: str = None, vnfs: list = None, throughput: float = None, latency: float = None, percentage_traffic: float = None, availability: float = None, source: Location = None, sink: Location = None):
        self.description = description
        self.vnfs = vnfs
        self.throughput = throughput
        self.latency = latency
        self.percentage_traffic = percentage_traffic
        self.availability = availability
        self.source = source
        self.sink = sink
        self.graph = None
        self.status = False
    
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
    
    def get_vnfs(self, vnfs):
        """
        Given a list of VNFs, returns the list of required VNFs that match the name of the 
        """
        to_return = []
        for v in self.vnfs:
            for vnf in vnfs:
                if vnf.description == v:
                    to_return.append(vnf)
        return to_return

    def to_json(self) -> dict:
        """
        Returns a json dictionary describing the vnf.
        """
        to_return = {}
        to_return["name"] = self.description
        to_return["vnfs"] = self.vnfs
        to_return["throughput"] = self.throughput
        to_return["latency"] = self.latency
        to_return["percentage_traffic"] = self.percentage_traffic
        to_return["availability"] = self.availability
        if self.source != None:
            to_return["source"] = self.source.to_json()
        if self.sink != None:
            to_return["sink"] = self.sink.to_json()
        if self.graph != None:
            to_return["paths"] = [p.to_json() for p in self.graph.paths]
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
    
    def load_from_json(self, filename: str):
        """
        Given a json file, it loads the data and stores as instance of VNF.
        """
        if filename[-5:] != ".json":
            filename = filename + ".json"

        # Opens the json and extracts the data
        with open(filename) as f:
            data = json.load(f)
        
        self.description = data["name"]
        self.vnfs = data["vnfs"]
        self.throughput = data["throughput"]
        self.latency = data["latency"]
        self.percentage_traffic = data["percentage_traffic"]
        self.availability = data["availability"]

    def make_graph(self, vnfs: list, topology: Network):
        """
        Makes a graph representing the service in the network
        """
        required_vnfs = self.get_vnfs(vnfs)
        n_layers = len(required_vnfs) + 1
        layers = []
        nodes, edges = [], []
        # Makes copy of network into n_components + 1 layers
        for i in range(n_layers):
            layer = topology.copy(topology.description + "_l{}".format(i))
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
        serv_g = service_graph(self.description + "_graph", nodes, edges, topology, self, n_layers)
        nodes = [n for n in topology.locations if isinstance(n, Node) == True]

        for node in nodes:
            for l in range(n_layers - 1):
                from_ = serv_g.get_location_by_description(node.description + "_l{}".format(l))
                to_ = serv_g.get_location_by_description(node.description + "_l{}".format(l+1))
                serv_g.add_link(Link(from_, to_, latency = required_vnfs[l].latency, cost = 0, assignment_link = True))

        self.graph = serv_g

        # Initialises path using dummy node.
        used_edges, used_nodes = [], []
        # Gets nodes used in initial path.
        used_nodes.append(self.graph.get_location_by_description(self.source.description + "_l0"))
        used_nodes.append(self.graph.get_location_by_description(self.sink.description + "_l{}".format(n_layers-1)))
        for l in range(n_layers):
            used_nodes.append(self.graph.get_location_by_description("dummy_l{}".format(l)))
        
        for edge in self.graph.links:
            # Start link from source to dummy.
            if edge.source.description == self.source.description + "_l0" and edge.sink.description == "dummy_l0":
                used_edges.append(edge)
            # End link from dummy to sink.
            if edge.source.description == "dummy_l{}".format(n_layers-1) and edge.sink.description == self.sink.description + "_l{}".format(n_layers-1):
                used_edges.append(edge)
            # Edges between dummy layers:
            if "dummy" in edge.source.description and "dummy" in edge.sink.description:
                used_edges.append(edge)

        path = service_path(self.description, used_nodes, used_edges, topology, self, n_layers = n_layers)
        self.graph.add_path(path)
        return path