import json
from xml.dom.minicompat import NodeList
from topology.location import *
from topology.link import *
from topology.network import Network
import random
import xmltodict
from math import sqrt
import networkx as nx

def parse_network_from_xml(filename: str, name: str, output_dir: str, n_core_dc: int, n_edge_dc: int, max_latency = 2, bandwidths = [10000, 40000, 100000]):
    """
    This parses an xml file and makes a Network object then saves the network object as a json.

    Params
    -----------------------------
        filename:       str
                            name of the .xml file
        name:           str
                            name of the output json network
        output_dir:     str
                            name of directory to save json to
        n_core_dc:      int
                            number of core DCs
        n_edge_dc:      int
                            number of edge DCs
        max_latency:    int
                            max latency used to scale latency of edges.
        bandwidths:     list[int]
                            bandwidth to use for edges.
    Returns
    -----------------------------
                            None
    """
    # Opens the .xml file
    with open(filename, "r") as f:
        data = f.read()
    data = xmltodict.parse(data)

    # Gets node and link attributes.
    vertices = data['network']['networkStructure']['nodes']['node']
    edges = data["network"]["networkStructure"]["links"]["link"]
    assert n_core_dc + n_edge_dc <= len(vertices), "Number of DC's must be less than total number of vertices."

    # Makes network x object to calculate betweeness centrality. This is used to find most important nodes for core DC.
    G = nx.Graph()
    distances = []
    for v in vertices:
        G.add_node(v["@id"])
    for e in edges:
        source, sink = e["source"], e["target"]
        for vertice in vertices:
            if vertice["@id"] == source:
                source_x, source_y = float(vertice["coordinates"]["x"]), float(vertice["coordinates"]["y"])
            if vertice['@id'] == sink:
                sink_x, sink_y = float(vertice["coordinates"]["x"]), float(vertice["coordinates"]["y"])
        distance = sqrt((sink_y - source_y)**2 + (sink_x - source_x)**2)
        distances.append(distance)
        G.add_edge(e["source"], e["target"], weight = distance)
    bc = nx.betweenness_centrality(G, weight = "weight")
    
    max_distance = max(distances) 

    # Sorts by betweeness centrality and makes the core DCs the ones with highest betweeness centrality.
    sorted_nodes = sorted(bc, reverse = True, key = bc.get)
    core_nodes = sorted_nodes[:n_core_dc]
    other_nodes = sorted_nodes[n_core_dc:]
    random.shuffle(other_nodes)
    # Randomly samples n_edge_dc from other nodes. All remaining nodes are switches.
    edge_nodes = other_nodes[:n_edge_dc]
    switches = other_nodes[n_edge_dc:]

    locations, links = [], []
    # Makes network object.

    for v in vertices:
        # If its a switch.
        if v["@id"] in switches:
            locations.append(Switch(v["@id"] + "Switch"))
        # Elif it is a edge node.
        elif v["@id"] in edge_nodes:
            locations.append(Node(v["@id"] + "EdgeNode", 40, 40, availability=0.9999))
        # Else if its a core DC, we add a gateway switch and 3 nodes.
        else:
            g = Switch(v["@id"] + "Gateway")
            n1 = Node(v["@id"] + "CoreNode1", 100, 100, availability=0.9999, handles_requests=False)
            n2 = Node(v["@id"] + "CoreNode2", 100, 100, availability=0.9999, handles_requests=False)
            n3 = Node(v["@id"] + "CoreNode3", 100, 100, availability=0.9999, handles_requests=False)
            locations += [g, n1, n2, n3]
            # Adds the edges between the gateway and nodes.
            links.append(Link(g, n1))
            links.append(Link(g, n2))
            links.append(Link(g, n3))
    
    for e in edges:
        source_id, sink_id = e["source"], e["target"]
        bandwidth = random.choice(bandwidths)
        source, sink = None, None
        # Gets the coordinates of source and sink to compute latency.
        for v in vertices:
            if v["@id"] == source_id:
                source_x, source_y = float(v["coordinates"]["x"]), float(v["coordinates"]["y"])
            if v['@id'] == sink_id:
                sink_x, sink_y = float(v["coordinates"]["x"]), float(v["coordinates"]["y"])
        latency = max_latency * sqrt((sink_y - source_y)**2 + (sink_x - source_x)**2)/max_distance
        for location in locations:
            if location.description in [source_id + "Switch", source_id + "EdgeNode", source_id + "Gateway"]:
                source = location
            elif location.description in [sink_id + "Switch", sink_id + "EdgeNode", sink_id + "Gateway"]:
                sink = location
        links.append(Link(source, sink, float(bandwidth), latency))    
                
    network = Network(name, locations, links)
    network.save_as_json(output_dir + name)

def main():
    """
    Script to load an xml file, translate to network object and save as json.
    """
    filename = "data_used/network_xmls/nobeleu.xml"
    output_dir = "data_used/networks/"
    name = "nobeleu"
    parse_network_from_xml(filename, name, output_dir, 5, 15)

if __name__ == "__main__":
    main()