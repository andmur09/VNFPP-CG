from typing import List

from kiwisolver import strength
from topology.location import Location
from topology.location import Dummy
from topology.link import Link
from topology.location import Node
import json
from topology.numpy_encoder import NpEncoder
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
    def __init__(self, description: str, locations: list, links: list):
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
    
    def get_link_by_locations(self, source_id, sink_id) -> Link:
        """
        returns an edge given two location ids.
        """
        for link in self.links:
            if link.source.id == source_id and link.sink.id == sink_id:
                return link
            elif link.sink.id == source_id and link.source.id == sink_id:
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
            self.add_link(Link(dummy, l))
            self.add_link(Link(l, dummy))

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
    
    def print(self):
        """
        Prints information about the network.
        TODO: This needs updating.
        """
        for link in self.links:
            print("\n")
            print("Description: ", link.get_description())
            print("Latency: {}, Bandwidth: {}".format(link.latency, link.bandwidth))
            print("Source: ", link.source.description)
            print("\tType: ", link.source.type)
            if link.source.type == "node":
                print("\tCPU: ", link.source.cpu)
                print("\tRAM: ", link.source.ram)
                print("\tCost: ", link.source.cost)
            print("Sink: ", link.sink.description)
            print("\tType: ", link.sink.type)
            if link.sink.type == "node":
                print("\tCPU: ", link.sink.cpu)
                print("\tRAM: ", link.sink.ram)
                print("\tCost: ", link.sink.cost)
    
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