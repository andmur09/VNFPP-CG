from topology.datacenter import Datacenter
from topology.location import Location
import itertools
import graphviz as gvz

class service_graph(Datacenter):
    """
    This class is used to make a service graph. Service graph can be used for optimising datacenter.
    """
    def __init__(self, name, locations, links):
        super().__init__(name, locations, links)
        self.paths = [] 

    def add_path(self, path):
        """
        Adds a path to the list of found paths.
        """
        self.paths.append(path)
    
    def get_start_node(self):
        """
        Gets the start node in the graph, i.e. the one with no incoming edges.
        """
        for i in self.locations:
            incoming = [l for l in self.links if l.sink == i]
            if not incoming:
                return i
    
    def get_end_node(self):
        """
        Gets the end node in the graph, i.e. the one with no outgoing edges.
        """
        for i in self.locations:
            outgoing = [l for l in self.links if l.source == i]
            if not outgoing:
                return i

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

class service_path(Datacenter):
    """
    This class represents a path on the above graph class.
    """
    id_iter = itertools.count()
    def __init__(self, name, locations, links, times_traversed, component_assignment):
        super().__init__(name, locations, links)
        self.name = name + str(next(Location.id_iter))
        self.times_traversed = times_traversed
        self.component_assignment = component_assignment
    
    def check_if_same(self, other_path):
        if self.times_traversed != other_path.times_traversed and self.component_assignment != other_path.component_assignment:
            return False
        else:
            return True
