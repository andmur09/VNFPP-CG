from topology.location import Location
inf = 1e6

class Link(object):
    """
    Class representing an edge (link) between two locations in the datacenter
    -------------------
    Params:
        source:             topology.location.Location
                                location object of source node.
        sink:               topology.location.Location
                                location object of sink node.
        bandwidth:          float
                                bandwidth of link.
        latency:            float
                                latency of link.
        cost:               float
                                cost of using link (modelled using dual variables).
        assignment_link:    bool
                                True is link is an assignment link between the same node in two layers of a graph, else False.
    """

    def __init__(self, source: Location = None, sink: Location = None, bandwidth: float = 1e6, latency: float = 0, cost = 0, assignment_link: bool = False):
        self.source = source
        self.sink = sink
        self.bandwidth = bandwidth
        self.latency = latency
        self.cost = cost
        self.assignment_link = assignment_link

    def copy(self):
        """
        makes a copy of the link
        """
        return Link(self.source.copy(), self.sink.copy(), self.bandwidth, self.latency)
    
    def copy_with_new_nodes(self, source: Location, sink: Location):
        """
        makes a copy of the link but with new source and sink nodes.
        """
        return Link(source, sink, self.bandwidth, self.latency)

    def get_description(self):
        """
        Gets the description of the edge in string format.
        """
        return "({}, {})".format(self.source.description, self.sink.description)
    
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
        return {"source": self.source.id, "sink": self.sink.id, "bandwidth": self.bandwidth, "latency": self.latency}
