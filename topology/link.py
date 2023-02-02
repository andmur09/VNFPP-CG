import copy
from topology.location import Location

class Link(object):
    """
    Class representing an edge (link) between two locations in the datacenter
        \param description    String name describing link
        \param soure          Start node in the link (instance of location class)
        \param sink           End node in the link (instance of location class)
        \param bandwidth      Float bandwidth of link
        \param latency      Float bandwidth of link
        \param cost      Float bandwidth of link
        \param biderectional  Boolean flag for leaf-leaf links which can flow either way.
    """

    def __init__(self, source: Location, sink: Location, bandwidth: float, latency: float = 0, cost: float = 1, two_way: bool = False):
        self.source = source
        self.sink = sink
        self.description = "({}, {})".format(source.description, sink.description)
        self.bandwidth = bandwidth
        self.latency = latency
        self.cost = cost
        self.two_way = two_way

    def copy(self):
        """
        makes a copy of the link
        """
        return Link(self.source.copy(), self.sink.copy(), self.bandwidth, self.latency, self.cost, self.two_way)
    
    def copy_with_new_nodes(self, source: Location, sink: Location):
        """
        makes a copy of the link but with new source and sink nodes.
        """
        return Link(source, sink, self.bandwidth, self.latency, self.cost, self.two_way)
    
    def get_opposing_edge_description(self) -> str:
        """
        Gets the description of the edge representing sink -> source.
        """
        return "({}, {})".format(self.sink.description, self.source.description)
    
    def __str__(self) -> str:
        """
        string representation of link
        """
        return "Link {} ({}) -> {} ({})".format(self.source.description, self.source.id, self.sink.description, self.sink.id)

    def to_json(self) -> dict:
        """
        returns the link as a dictionary for use with json
        """
        return {"source": self.source.id, "sink": self.sink.id, "description": self.description, "bandwidth": self.bandwidth, "latency": self.latency, "cost": self.cost, "two_way": self.two_way}