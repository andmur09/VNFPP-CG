from service_class.component import Component
from service_class.graph import service_graph
from topology.datacenter import Datacenter
from topology.link import Link
from topology.location import *

inf = 1000000000
eps = 1e-9

class Service(object):
    """
    Class representing a service
        \param description          String description of component
        \param components           List of components used in service
        \param required_latency     Float defining the latency required for the service
        \param required throughput  Float defining the throughput required for the service
    """
    def __init__(self, description: str, components: list, required_throughput: float, required_latency: float):
        self.description = description
        self.components = components
        self.required_throughput = required_throughput
        self.required_latency = required_latency
        self.graphs = {}
    
    def add_component(self, component: Component):
        """
        Adds a component to the service.
        """
        self.components.append(component)
    
    def get_graph(self, topology):
        """
        Given a topology, it finds the equivalent service graph and returns if found.
        """
        try:
            return self.graphs[topology.name]
        except KeyError:
            print("No graph for given topology. Try using service.addGraph(topology) to create one")
            return None
        
    def print(self):
        """
        Prints the service in a readable format.
        """
        print("\n")
        print("Description: ", self.description)
        print("Required Latency: ", self.required_latency)
        print("Required Throughput: ", self.required_throughput)
        for component in self.components:
            print("Component: ", component.description)
            print("\tRequired CPU: ", component.required_cpu)
            print("\tRequired CPU: ", component.required_ram)
        print("\n")
    
    def add_graph(self, topology: Datacenter):
        """
        Makes a graph representing the service in the datacenter
        """
        layers = topology.get_locations_by_types()
        no_components = len(self.components)
        graph_segments = {}

        for k in range(no_components+1):
            if k == 0:
                new_nodes = []
                new_edges = []
                # For the initial layer representing gateway to component1
                # Makes a copy of all nodes in all layers
                for layer in layers:
                    for node in layer:
                        new_nodes.append(node.copy())
                # Makes a copy of all links in the topology
                for edge in topology.links:    
                    for source in new_nodes:
                        for sink in new_nodes:
                            if edge.source.description == source.description and edge.sink.description == sink.description:
                                new_edge = edge.copy_with_new_nodes(source, sink)
                                new_edge.cost = 1
                                new_edges.append(new_edge)
                                if edge.two_way == True:
                                    new_edge = edge.copy_with_new_nodes(sink, source)
                                    new_edge.cost = 1
                                    new_edges.append(new_edge)
                graph_segments[str(k)] = (new_nodes, new_edges)

            elif k == no_components:
                new_nodes = []
                new_edges = []
                # For the final layer representing last component to gateway
                for layer in layers:
                    for node in layer:
                        new_nodes.append(node.copy())
                for edge in topology.links:
                    for source in new_nodes:
                        for sink in new_nodes:
                            if edge.source.description == sink.description and edge.sink.description == source.description:
                                new_edge = edge.copy_with_new_nodes(source, sink)
                                new_edge.cost = 1
                                new_edges.append(new_edge)
                                if edge.two_way == True:
                                    new_edge = edge.copy_with_new_nodes(sink, source)
                                    new_edge.cost = 1
                                    new_edges.append(new_edge)
                graph_segments[str(k)] = (new_nodes, new_edges)
            else:
                # For inter component layers:
                new_nodes = []
                new_edges = []

                out_nodes = []
                in_nodes = []
                mid_nodes = []

                intermediate_layers = layers[2:]
                for layer in intermediate_layers:
                    for node in layer:
                        # Out are those for transport from component one, up the layers
                        out_nodes.append(node.copy())
                        # In nodes are from the top layer down to component two
                        in_nodes.append(node.copy())

                for node in layers[1]:
                    # mid nodes are nodes in top layer (one down from gateway)
                    mid_nodes.append(node)

                for edge in topology.links:
                    # Adds links between mid layers
                    for source in out_nodes:
                        for sink in out_nodes:
                            if edge.source.description == sink.description and edge.sink.description == source.description:
                                new_edge = edge.copy_with_new_nodes(source, sink)
                                new_edge.cost = 1
                                new_edges.append(new_edge)
                                if edge.two_way == True:
                                    new_edge = edge.copy_with_new_nodes(sink, source)
                                    new_edge.cost = 1
                                    new_edges.append(new_edge)
                    for source in in_nodes:
                        for sink in in_nodes:
                            if edge.source.description == source.description and edge.sink.description == sink.description:
                                new_edge = edge.copy_with_new_nodes(source, sink)
                                new_edge.cost = 1
                                new_edges.append(new_edge)
                                if edge.two_way == True:
                                    new_edge = edge.copy_with_new_nodes(sink, source)
                                    new_edge.cost = 1
                                    new_edges.append(new_edge)
                    # Adds links between out layer and mid
                    for source in out_nodes:
                        for sink in mid_nodes:
                            if edge.source.description == sink.description and edge.sink.description == source.description:
                                new_edge = edge.copy_with_new_nodes(source, sink)
                                new_edge.cost = 1
                                new_edges.append(new_edge)
                                if edge.two_way == True:
                                    new_edge = edge.copy_with_new_nodes(sink, source)
                                    new_edge.cost = 1
                                    new_edges.append(new_edge)
                    # # Adds links between mid layer and in
                    for source in mid_nodes:
                        for sink in in_nodes:
                            if edge.source.description == source.description and edge.sink.description == sink.description:
                                new_edge = edge.copy_with_new_nodes(source, sink)
                                new_edge.cost = 1
                                new_edges.append(new_edge)
                                if edge.two_way == True:
                                    new_edge = edge.copy_with_new_nodes(sink, source)
                                    new_edge.cost = 1
                                    new_edges.append(new_edge)
                # Adds links between out version of node and in version of same node to prevent having to go up a layer
                for source in out_nodes:
                    for sink in in_nodes:
                        if source.description == sink.description:
                            new_edges.append(Link(source, sink, inf, latency = 0, cost = 0, two_way=False))
                new_nodes = out_nodes + mid_nodes + in_nodes

                graph_segments[str(k)] = (new_nodes, new_edges)

        #print([l.type for l in topology.locations])
        # # Joins segments of graph with nodes representing assignment of component to node.
        for i in range(no_components):
            new_nodes = []
            new_edges = []
            for l in topology.locations:
                # Adds dummy nodes representing node_component[i]
                if l.type == "Node":
                    new_nodes.append(Node(l.description + "[" + self.components[i].description + "]", "dummy", inf, inf))
        
            # Joins pre layer to nodes
            pre_layer = graph_segments[str(i)]
            for source in pre_layer[0]:
                for sink in new_nodes:
                    # Last part checks that there is no incoming edges to the source node
                    #if sink.description == source.description + "_" + _service.components[i].description:
                    if sink.description == source.description + "[" + self.components[i].description + "]" and not [l for l in pre_layer[1] if l.source == source]:
                        new_edges.append(Link(source, sink, {"bandwidth": inf, "latency": 0, "cost": 0}, two_way=False))
                
            # # Joins post layer to nodes
            post_layer = graph_segments[str(i+1)]
            for source in new_nodes:
                for sink in post_layer[0]:
                    # Last part checks that there is no outgoing edges to the sink node
                    if source.description == sink.description + "[" + self.components[i].description + "]" and not [l for l in post_layer[1] if l.sink == sink]:
                    #if sink.description == source.description + "_" + _service.components[i].description and not [l for l in post_layer[1] if l.sink == sink]:
                        new_edges.append(Link(source, sink, {"bandwidth": inf, "latency": 0, "cost": 0}, two_way=False))
                
            graph_segments["Component {}".format(i)] = (new_nodes, new_edges)

        nodes = []
        edges = []
        for key in graph_segments:
            nodes += graph_segments[key][0]
            edges += graph_segments[key][1]
        self.graphs[topology.description] = service_graph(topology.description + "_" + self.description, nodes, edges)