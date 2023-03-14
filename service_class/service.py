from lib2to3.pgen2.token import PERCENTEQUAL
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
    def __init__(self, description: str, vnfs: list, throughput: float, latency: float, percentage_traffic: float, availability: float = None, source: Location = None, sink: Location = None):
        self.description = description
        self.vnfs = vnfs
        self.throughput = throughput
        self.latency = latency
        self.availability = availability
        self.percentage_traffic = percentage_traffic
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
    
    def get_vnfs(self, vnfs):
        """
        Given a list of VNFs, returns the list of required VNFs that match the name of the 
        """
        return [v for v in vnfs if v.description in self.vnfs]

    def to_json(self) -> dict:
        """
        Returns a json dictionary describing the vnf.
        """
        return {"name": self.description, "vnfs": self.vnfs, "throughput": self.throughput, "latency": self.latency, "percentage_traffic": self.percentage_traffic, "availability": self.availability}

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
        if json_file[-5:] != ".json":
            json_file = json_file + ".json"

        # Opens the json and extracts the data
        with open(json_file) as f:
            data = json.load(f)
        
        assert data.keys() == ["name", "cpu", "ram", "throughput", "latency", "availability"], "Keys in JSON don't match expected input for type vnf."
        self.description, self.cpu, self.ram, self.throughput, self.latency, self.availability = data["name"], data["cpu"], data["ram"], data["throughput"], data["latency"], data["availability"]
    
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
            for node in layer.locations:
                node.description = node.description + "_l{}".format(i)
            nodes += layer.locations
            edges += layer.links
            layers.append(layer)
        
        # Adds edges connecting the nodes on layer l to layer l+1. Traversing this edge represents assigning component l to the node on layer l.
        serv_g = service_graph(self.description + "_graph", nodes, edges, topology, self, n_layers)
        nodes = [n for n in topology.locations if isinstance(n, Node) == True]

        for node in nodes:
            for l in range(n_layers - 1):
                from_ = serv_g.get_location_by_description(node.description + "_l{}".format(l))
                to_ = serv_g.get_location_by_description(node.description + "_l{}".format(l+1))
                serv_g.add_link(Link(from_, to_, latency = self.vnfs[l].latency, cost = 0, assignment_link = True))

        self.graph = serv_g
        
        # Initialises path using dummy node.#
        
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
        path.save_as_dot()
        return path