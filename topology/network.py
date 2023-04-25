from topology.location import Location
from topology.location import Dummy
from topology.location import Node, Switch
import json
import graphviz as gvz

inf = 1000000
eps = 1e-6

class Network(object):
    """
    Class representing a network
        \param description    String name describing network
        \param locations      List of locations in the network (vertices)
        \param links          List of links in the network (edges)
    """
    def __init__(self, description: str = None, locations: list = None, links: list = None):
        self.description = description
        self.locations = locations
        self.links = links

    def copy(self, description: str):
        """
        returns a copy of the network.
        """
        nodes = [n.copy() for n in self.locations]
        edges = []
        for link in self.links:
            for node in nodes:
                if node.description == link.source.description:
                    source = node
                elif node.description == link.sink.description:
                    sink = node
            edges.append(link.copy_with_new_nodes(source, sink))
        return Network(description, nodes, edges)
  
    def get_location_by_description(self, description: str) -> Location:
        """
        returns a location in the network matching the description argument.
        """
        for location in self.locations:
            if location.description == description:
                return location
        return None

    def get_link_by_description(self, description: str) -> Link:
        """
        returns a link in the network mathching the description argument.
        """
        for link in self.links:
            if link.get_description() == description:
                return link
        return None
    
    def get_link_by_location(self, source_id, sink_id) -> Link:
        """
        returns an edge given two location ids.
        """
        for link in self.links:
            if link.source.id == source_id and link.sink.id == sink_id:
                return link
            elif link.sink.id == source_id and link.source.id == sink_id:
                return link
        return None

    def get_link_by_location_description(self, source, sink) -> Link:
        """
        returns an edge given two location ids.
        """
        for link in self.links:
            if link.source.description == source.description and link.sink.description == sink.description:
                return link
            elif link.sink.id == source.description and link.source.id == sink.description:
                return link
        return None

    def get_locations_by_type(self, type: str) -> list:
        """
        Gets a list of locations in the network matching a certain type.
        """
        if type not in ["Node", "Switch"]:
            raise ValueError("Invalid type. Type should be either Node or Switch.")
        return [i for i in self.locations if i.type == type]

    def outgoing_edge(self, location: Location) -> list:
        """
        Returns a list of outgoing links from a particular location
        """
        return [l for l in self.links if l.source == location]
    
    def incoming_edge(self, location: Location) -> list:
        """
        Returns a list of incoming links from a particular location
        """
        return [l for l in self.links if l.sink == location]

    def add_link(self, link: Link) -> None:
        """
        Adds a link to the network.
        """
        self.links.append(link)

    def add_location(self, location: Location) -> None:
        """
        Adds a location to the network.
        """
        self.locations.append(location)
    
    def add_dummy_node(self) -> None:
        """
        Adds a dummy node to the network. See topology.location.Dummy for details.
        """
        dummy = Dummy("dummy")
        self.add_location(dummy)
        other_locations = [l for l in self.locations if l != dummy]
        for l in other_locations:
            self.add_link(Link(dummy, l, latency = 0, bandwidth=inf))
            self.add_link(Link(l, dummy, latency = 0, bandwidth=inf))

    def to_json(self) -> dict:
        """
        Returns a json dictionary describing the network.
        """
        to_return = {"name": self.description, "locations": [], "links": []}
        for location in self.locations:
            to_return["locations"].append(location.to_json())
        for link in self.links:
            to_return["links"].append(link.to_json())
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

    def load_from_json(self, filename):
        """
        Given a json file, it loads the data and stores as instance of the network.
        """
        if filename[-5:] != ".json":
            filename = filename + ".json"

        # Opens the json and extracts the data
        with open(filename) as f:
            data = json.load(f)
        
        assert list(data.keys()) == ["name", "locations", "links"], "Keys in JSON don't match expected input for type network."

        self.description, self.locations, self.links = data["name"], [], []
        locations, links = data["locations"], data["links"]
        # Adds locations
        for location in locations:
            if location["type"] == "Switch":
                toAdd = Switch()
                toAdd.load_from_dict(location)
                self.locations.append(toAdd)
            elif location["type"] == "Node":
                toAdd = Node()
                toAdd.load_from_dict(location)
                self.locations.append(toAdd)
        
        # Adds links by matching id's of newly created locations.
        for link in links:
            assert list(link.keys()) == ["source", "sink", "bandwidth", "latency"], "Keys in JSON don't match expected input for type link."
            toAdd = Link()
            for location in self.locations:
                if location.id == link["source"]:
                    toAdd.source = location
                elif location.id == link["sink"]:
                    toAdd.sink = location
            if toAdd.source == None or toAdd.sink == None:
                print(link)
                raise ValueError("No source or sink found with that id.")
            toAdd.bandwidth = link["bandwidth"]
            toAdd.latency = link["latency"]
            self.links.append(toAdd)

    def __str__(self):
        """
        Prints the DC network to string.
        """
        to_return = "Name:\n"
        to_return += "\t{}\n".format(self.description)
        to_return += "Nodes:\n"
        for location in self.locations:
            to_return += "\t" + str(location.to_json()) + "\n"
        to_return += "Edges:\n"
        for link in self.links:
            to_return += "\t" + str(link.to_json()) + "\n"
        return to_return
    
    def save_as_dot(self, filename = None):
        """
        saves the network topology as a DOT to filename.dot
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
            plot.edge(str(link.source.id), str(link.sink.id))

        with open(filename, "w") as f:
            f.write(plot.source)