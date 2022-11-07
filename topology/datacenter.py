from typing import List
from topology.location import Location
from topology.link import Link
from topology.location import Node
import json
from topology.numpy_encoder import NpEncoder


class Datacenter(object):
    """
    Class representing a datacenter
        \param description    String name describing datacenter
        \param locations      List of locations in the datacenter (vertices)
        \param links          List of links in the datacenter (edges)
    """
    def __init__(self, description: str, locations: list, links: list):
        self.description = description
        self.locations = locations
        self.links = links

    def copy(self, name):
        """
        returns a copy of the datacenter
        """
        return Datacenter(name,  self.locations[:], [link.copy() for link in self.links])
    
    def get_location_by_description(self, description: str) -> Location:
        """
        returns a location in the datacenter matching the description argument.
        """
        for location in self.locations:
            if location.description == description:
                return location
        return None
    
    def get_edge_by_locations(self, source_id, sink_id) -> Link:
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
        Gets a list of locations in the datacenter matching a certain type.
        """
        if type not in ["Gateway", "SuperSpine", "Spine", "Leaf", "Node", "Dummy"]:
            raise ValueError("Invalid type. Type should be in [Gateway, SuperSpine, Spine, Leaf, Node, Dummy]")
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
        return [l for l in self.getLinks() if l.sink == location]

    def add_link(self, link: Link) -> None:
        """
        Adds a link to the datacenter.
        """
        self.links.append(link)

    def add_location(self, location: Location) -> None:
        """
        Adds a location to the datacenter.
        """
        self.locations.append(location)

    def to_json(self) -> dict:
        """
        Returns a json dictionary describing the datacenter.
        """
        to_return = {"name": self.description, "locations": [], "links": []}
        for location in self.locations:
            to_return["locations"].append(location.to_json())
        for link in self.links:
            to_return["locations"].append(link.to_json())
        return to_return

    def save_as_json(self, filename = None):
        """
        saves the datacenter as a JSON to filename.json
        """
        to_dump = self.to_json
        if filename != None:
            if filename[-5:] != ".json":
                filename = filename + ".json"
        else:
            filename = self.description + ".json"

        with open(filename, 'w') as fp:
            json.dump(to_dump, fp, indent=4, separators=(", ", ": "), cls=NpEncoder)
    
    def print(self):
        """
        Prints information about the datacenter.
        """
        for link in self.getLinks():
            print("\n")
            print("Description: ", link.description)
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

    # def addArtificialNode(self):
    #     # Adds an artifical sink node connected to each node in list 'connections'. If connections contains one location 'B' then it will add a link B -> Sink
    #     dummy = Node("Dummy", 10000, 10000, 10000)
    #     self.addLocation(dummy)
    #     self.addLink(self.getLocationsByType("gateway")[0], dummy, {"bandwidth": 10000, "latency": 0, "cost": 10000})