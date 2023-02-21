import random
from service_class.vnf import Component
from service_class.service import Service
from topology.network import Network
from topology.link import Link
from topology.location import *

def make_datacenter():
    """
    Makes a copy of the datacenter with random node costs.
    """
    # Makes a list of the physical locations in the datacenter

    # Datacenter Gateway
    g = Gateway("Gateway")
    gs = [g]

    # Super Spine switches
    ss1 = SuperSpine("SuperSpine1")
    ss2 = SuperSpine("SuperSpine2")
    ss3 = SuperSpine("SuperSpine3")
    ss4 = SuperSpine("SuperSpine4")
    sss = [ss1, ss2, ss3, ss4]

    # Spine switches
    s1 = Spine("Spine1")
    s2 = Spine("Spine2")
    s3 = Spine("Spine3")
    s4 = Spine("Spine4")
    s5 = Spine("Spine5")
    s6 = Spine("Spine6")
    s7 = Spine("Spine7")
    s8 = Spine("Spine8")
    ss = [s1, s2, s3, s4, s5, s6, s7, s8]

    # Leaf switches
    l1 = Leaf("Leaf1")
    l2 = Leaf("Leaf2")
    l3 = Leaf("Leaf3")   
    l4 = Leaf("Leaf4")
    l5 = Leaf("Leaf5")
    l6 = Leaf("Leaf6")
    l7 = Leaf("Leaf7")
    l8 = Leaf("Leaf8")
    l9 = Leaf("Leaf9")
    l10 = Leaf("Leaf10")
    l11 = Leaf("Leaf11")   
    l12 = Leaf("Leaf12")
    l13 = Leaf("Leaf13")
    l14 = Leaf("Leaf14")
    l15 = Leaf("Leaf15")
    l16 = Leaf("Leaf16")
    ls = [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16]

    # Nodes
    n1 = Node("Node1", 10, 64, random.randint(1, 10))
    n2 = Node("Node2", 10, 64, random.randint(1, 10))
    n3 = Node("Node3", 10, 64, random.randint(1, 10))
    n4 = Node("Node4", 10, 64, random.randint(1, 10))
    n5 = Node("Node5", 10, 64, random.randint(1, 10))
    n6 = Node("Node6", 10, 64, random.randint(1, 10))
    n7 = Node("Node7", 10, 64, random.randint(1, 10))
    n8 = Node("Node8", 10, 64, random.randint(1, 10))
    n9 = Node("Node9", 10, 64, random.randint(1, 10))
    n10 = Node("Node10", 10, 64, random.randint(1, 10))
    n11 = Node("Node11", 10, 64, random.randint(1, 10))
    n12 = Node("Node12", 10, 64, random.randint(1, 10))
    n13 = Node("Node13", 10, 64, random.randint(1, 10))
    n14 = Node("Node14", 10, 64, random.randint(1, 10))
    n15 = Node("Node15", 10, 64, random.randint(1, 10))
    n16 = Node("Node16", 10, 64, random.randint(1, 10))
    ns = [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, n16]

    locations = gs + sss + ss + ls + ns

    # Makes a list of edges in the datacenter:
    edges = []
    # Gateway super spines
    edges.append(Link(g, ss1, 100, 1))
    edges.append(Link(g, ss2, 100, 1))
    edges.append(Link(g, ss3, 100, 1))
    edges.append(Link(g, ss4, 100, 1))
    # Super spines to spines
    edges.append(Link(ss1, s1, 100, 1))
    edges.append(Link(ss1, s2, 100, 1))
    edges.append(Link(ss1, s3, 100, 1))
    edges.append(Link(ss1, s4, 100, 1))
    edges.append(Link(ss2, s1, 100, 1))
    edges.append(Link(ss2, s2, 100, 1))
    edges.append(Link(ss2, s3, 100, 1))
    edges.append(Link(ss2, s4, 100, 1))
    edges.append(Link(ss3, s5, 100, 1))   
    edges.append(Link(ss3, s6, 100, 1))
    edges.append(Link(ss3, s7, 100, 1))  
    edges.append(Link(ss3, s8, 100, 1))    
    edges.append(Link(ss4, s5, 100, 1))   
    edges.append(Link(ss4, s6, 100, 1))
    edges.append(Link(ss4, s7, 100, 1))  
    edges.append(Link(ss4, s8, 100, 1))    
    # Spines to Leafs
    edges.append(Link(s1, l1, 40, 1))
    edges.append(Link(s1, l2, 40, 1))
    edges.append(Link(s1, l3, 40, 1))
    edges.append(Link(s1, l4, 40, 1))
    edges.append(Link(s1, l5, 40, 1))
    edges.append(Link(s1, l6, 40, 1))
    edges.append(Link(s1, l7, 40, 1))
    edges.append(Link(s1, l8, 40, 1))
    edges.append(Link(s2, l1, 40, 1))
    edges.append(Link(s2, l2, 40, 1))
    edges.append(Link(s2, l3, 40, 1))
    edges.append(Link(s2, l4, 40, 1))
    edges.append(Link(s2, l5, 40, 1))
    edges.append(Link(s2, l6, 40, 1))
    edges.append(Link(s2, l7, 40, 1))
    edges.append(Link(s2, l8, 40, 1))
    edges.append(Link(s3, l1, 40, 1))
    edges.append(Link(s3, l2, 40, 1))
    edges.append(Link(s3, l3, 40, 1))
    edges.append(Link(s3, l4, 40, 1))
    edges.append(Link(s3, l5, 40, 1))
    edges.append(Link(s3, l6, 40, 1))
    edges.append(Link(s3, l7, 40, 1))
    edges.append(Link(s3, l8, 40, 1))
    edges.append(Link(s4, l1, 40, 1))
    edges.append(Link(s4, l2, 40, 1))
    edges.append(Link(s4, l3, 40, 1))
    edges.append(Link(s4, l4, 40, 1))
    edges.append(Link(s4, l5, 40, 1))
    edges.append(Link(s4, l6, 40, 1))
    edges.append(Link(s4, l7, 40, 1))
    edges.append(Link(s4, l8, 40, 1))
    edges.append(Link(s5, l9, 40, 1))
    edges.append(Link(s5, l10, 40, 1))
    edges.append(Link(s5, l11, 40, 1))
    edges.append(Link(s5, l12, 40, 1))
    edges.append(Link(s5, l13, 40, 1))
    edges.append(Link(s5, l14, 40, 1))
    edges.append(Link(s5, l15, 40, 1))
    edges.append(Link(s5, l16, 40, 1))
    edges.append(Link(s6, l9, 40, 1))
    edges.append(Link(s6, l10, 40, 1))
    edges.append(Link(s6, l11, 40, 1))
    edges.append(Link(s6, l12, 40, 1))
    edges.append(Link(s6, l13, 40, 1))
    edges.append(Link(s6, l14, 40, 1))
    edges.append(Link(s6, l15, 40, 1))
    edges.append(Link(s6, l16, 40, 1))
    edges.append(Link(s7, l9, 40, 1))
    edges.append(Link(s7, l10, 40, 1))
    edges.append(Link(s7, l11, 40, 1))
    edges.append(Link(s7, l12, 40, 1))
    edges.append(Link(s7, l13, 40, 1))
    edges.append(Link(s7, l14, 40, 1))
    edges.append(Link(s7, l15, 40, 1))
    edges.append(Link(s7, l16, 40, 1))
    edges.append(Link(s8, l9, 40, 1))
    edges.append(Link(s8, l10, 40, 1))
    edges.append(Link(s8, l11, 40, 1))
    edges.append(Link(s8, l12, 40, 1))
    edges.append(Link(s8, l13, 40, 1))
    edges.append(Link(s8, l14, 40, 1))
    edges.append(Link(s8, l15, 40, 1))
    edges.append(Link(s8, l16, 40, 1))
    # Leaf to node links
    edges.append(Link(l1, n1, 1, 1))
    edges.append(Link(l1, n2, 1, 1))
    edges.append(Link(l2, n1, 1, 1))
    edges.append(Link(l2, n2, 1, 1))
    edges.append(Link(l3, n3, 1, 1))
    edges.append(Link(l3, n4, 1, 1))
    edges.append(Link(l4, n3, 1, 1))
    edges.append(Link(l4, n4, 1, 1))
    edges.append(Link(l5, n5, 1, 1))
    edges.append(Link(l5, n6, 1, 1))
    edges.append(Link(l6, n5, 1, 1))
    edges.append(Link(l6, n6, 1, 1))
    edges.append(Link(l7, n7, 1, 1))
    edges.append(Link(l7, n8, 1, 1))
    edges.append(Link(l8, n7, 1, 1))
    edges.append(Link(l8, n8, 1, 1))
    edges.append(Link(l9, n9, 1, 1))
    edges.append(Link(l9, n10, 1, 1))
    edges.append(Link(l10, n9, 1, 1))
    edges.append(Link(l10, n10, 1, 1))
    edges.append(Link(l11, n11, 1, 1))
    edges.append(Link(l11, n12, 1, 1))
    edges.append(Link(l12, n11, 1, 1))
    edges.append(Link(l12, n12, 1, 1))
    edges.append(Link(l13, n13, 1, 1))
    edges.append(Link(l13, n14, 1, 1))
    edges.append(Link(l14, n13, 1, 1))
    edges.append(Link(l14, n14, 1, 1))
    edges.append(Link(l15, n15, 1, 1))
    edges.append(Link(l15, n16, 1, 1))
    edges.append(Link(l16, n15, 1, 1))
    edges.append(Link(l16, n16, 1, 1))
    # Inter leaf edges
    edges.append(Link(l1, l2, 40, 1, two_way=True))
    edges.append(Link(l3, l4, 40, 1, two_way=True))
    edges.append(Link(l5, l6, 40, 1, two_way=True))
    edges.append(Link(l7, l8, 40, 1, two_way=True))
    edges.append(Link(l9, l10, 40, 1, two_way=True))
    edges.append(Link(l11, l12, 40, 1, two_way=True))
    edges.append(Link(l13, l14, 40, 1, two_way=True))
    edges.append(Link(l5, l6, 40, 1, two_way=True))

    return Network("EricssonDC", locations, edges)

def randomServices(no_services, no_components):
    components = []
    for i in range(no_components):
        components.append(Component("Component_{}".format(i), {"cpu": random.randint(1, 1), "ram": random.randint(1, 11)}, random.randint(1, 2)))
    services = []
    for i in range(no_services):
        random_components = random.choices(components, k=random.randint(1, 4))
        required_components = set()
        for j in random_components:
            required_components.add(j)
        required_components = list(required_components)
        services.append(Service("Service_{}".format(i), required_components, random.randint(1, 6), 1000))
    return services

dc = make_datacenter()
dc.save_as_dot()
