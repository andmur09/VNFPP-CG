import json
from topology.location import *
from topology.link import *
from topology.network import Network
import random

def main():
    # Makes a list of the physical locations in the network.

    ################ CORE DC ###################
    # Datacenter Gateway
    g1 = Switch("CoreGateway")
    locations = []

    # Super Spine switches
    ss1 = Switch("CoreSuperSpine1")
    ss2 = Switch("CoreSuperSpine2")
    ss3 = Switch("CoreSuperSpine3")
    ss4 = Switch("CoreSuperSpine4")
    

    # Spine switches
    s1 = Switch("CoreSpine1")
    s2 = Switch("CoreSpine2")
    s3 = Switch("CoreSpine3")
    s4 = Switch("CoreSpine4")
    s5 = Switch("CoreSpine5")
    s6 = Switch("CoreSpine6")
    s7 = Switch("CoreSpine7")
    s8 = Switch("CoreSpine8")

    # Leaf switches
    l1 = Switch("CoreLeaf1")
    l2 = Switch("CoreLeaf2")
    l3 = Switch("CoreLeaf3")   
    l4 = Switch("CoreLeaf4")
    l5 = Switch("CoreLeaf5")
    l6 = Switch("CoreLeaf6")
    l7 = Switch("CoreLeaf7")
    l8 = Switch("CoreLeaf8")
    l9 = Switch("CoreLeaf9")
    l10 = Switch("CoreLeaf10")
    l11 = Switch("CoreLeaf11")   
    l12 = Switch("CoreLeaf12")
    l13 = Switch("CoreLeaf13")
    l14 = Switch("CoreLeaf14")
    l15 = Switch("CoreLeaf15")
    l16 = Switch("CoreLeaf16")

    # Nodes
    n1 = Node("CoreNode1", 100, 640, availability=0.9999)
    n2 = Node("CoreNode2", 100, 640, availability=0.9999)
    n3 = Node("CoreNode3", 100, 640, availability=0.9999)
    n4 = Node("CoreNode4", 100, 640, availability=0.9999)
    n5 = Node("CoreNode5", 100, 640, availability=0.9999)
    n6 = Node("CoreNode6", 100, 640, availability=0.9999)
    n7 = Node("CoreNode7", 100, 640, availability=0.9999)
    n8 = Node("CoreNode8", 100, 640,availability=0.9999)
    n9 = Node("CoreNode9", 100, 640, availability=0.9999)
    n10 = Node("CoreNode10", 100, 640, availability=0.9999)
    n11 = Node("CoreNode11", 100, 640, availability=0.9999)
    n12 = Node("CoreNode12", 100, 640, availability=0.9999)
    n13 = Node("CoreNode13", 100, 640, availability=0.9999)
    n14 = Node("CoreNode14", 100, 640, availability=0.9999)
    n15 = Node("CoreNode15", 100, 640, availability=0.9999)
    n16 = Node("CoreNode16", 100, 640, availability=0.9999)

    locations = [g1, ss1, ss2, ss3, ss4, s1, s2, s3, s4, s5, s6, s7, s8,
                l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16,
                n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, n16]

    # Makes a list of edges in the datacenter:
    edges = []
    # Gateway super spines
    edges.append(Link(g1, ss1, 100000))
    edges.append(Link(g1, ss2, 100000))
    edges.append(Link(g1, ss3, 100000))
    edges.append(Link(g1, ss4, 100000))
    # Super spines to spines
    edges.append(Link(ss1, s1, 100000))
    edges.append(Link(ss1, s2, 100000))
    edges.append(Link(ss1, s3, 100000))
    edges.append(Link(ss1, s4, 100000))
    edges.append(Link(ss2, s1, 100000))
    edges.append(Link(ss2, s2, 100000))
    edges.append(Link(ss2, s3, 100000))
    edges.append(Link(ss2, s4, 100000))
    edges.append(Link(ss3, s5, 100000))   
    edges.append(Link(ss3, s6, 100000))
    edges.append(Link(ss3, s7, 100000))  
    edges.append(Link(ss3, s8, 100000))    
    edges.append(Link(ss4, s5, 100000))   
    edges.append(Link(ss4, s6, 100000))
    edges.append(Link(ss4, s7, 100000))  
    edges.append(Link(ss4, s8, 100000))    
    # Spines to Leafs
    edges.append(Link(s1, l1, 40000))
    edges.append(Link(s1, l2, 40000))
    edges.append(Link(s1, l3, 40000))
    edges.append(Link(s1, l4, 40000))
    edges.append(Link(s1, l5, 40000))
    edges.append(Link(s1, l6, 40000))
    edges.append(Link(s1, l7, 40000))
    edges.append(Link(s1, l8, 40000))
    edges.append(Link(s2, l1, 40000))
    edges.append(Link(s2, l2, 40000))
    edges.append(Link(s2, l3, 40000))
    edges.append(Link(s2, l4, 40000))
    edges.append(Link(s2, l5, 40000))
    edges.append(Link(s2, l6, 40000))
    edges.append(Link(s2, l7, 40000))
    edges.append(Link(s2, l8, 40000))
    edges.append(Link(s3, l1, 40000))
    edges.append(Link(s3, l2, 40000))
    edges.append(Link(s3, l3, 40000))
    edges.append(Link(s3, l4, 40000))
    edges.append(Link(s3, l5, 40000))
    edges.append(Link(s3, l6, 40000))
    edges.append(Link(s3, l7, 40000))
    edges.append(Link(s3, l8, 40000))
    edges.append(Link(s4, l1, 40000))
    edges.append(Link(s4, l2, 40000))
    edges.append(Link(s4, l3, 40000))
    edges.append(Link(s4, l4, 40000))
    edges.append(Link(s4, l5, 40000))
    edges.append(Link(s4, l6, 40000))
    edges.append(Link(s4, l7, 40000))
    edges.append(Link(s4, l8, 40000))
    edges.append(Link(s5, l9, 40000))
    edges.append(Link(s5, l10, 40000))
    edges.append(Link(s5, l11, 40000))
    edges.append(Link(s5, l12, 40000))
    edges.append(Link(s5, l13, 40000))
    edges.append(Link(s5, l14, 40000))
    edges.append(Link(s5, l15, 40000))
    edges.append(Link(s5, l16, 400003))
    edges.append(Link(s6, l9, 40000))
    edges.append(Link(s6, l10, 40000))
    edges.append(Link(s6, l11, 40000))
    edges.append(Link(s6, l12, 40000))
    edges.append(Link(s6, l13, 40000))
    edges.append(Link(s6, l14, 40000))
    edges.append(Link(s6, l15, 40000))
    edges.append(Link(s6, l16, 40000))
    edges.append(Link(s7, l9, 40000))
    edges.append(Link(s7, l10, 40000))
    edges.append(Link(s7, l11, 40000))
    edges.append(Link(s7, l12, 40000))
    edges.append(Link(s7, l13, 40000))
    edges.append(Link(s7, l14, 40000))
    edges.append(Link(s7, l15, 40000))
    edges.append(Link(s7, l16, 40000))
    edges.append(Link(s8, l9, 40000))
    edges.append(Link(s8, l10, 40000))
    edges.append(Link(s8, l11, 40000))
    edges.append(Link(s8, l12, 40000))
    edges.append(Link(s8, l13, 40000))
    edges.append(Link(s8, l14, 40000))
    edges.append(Link(s8, l15, 40000))
    edges.append(Link(s8, l16, 40000))
    # Leaf to node links
    edges.append(Link(l1, n1, 10000))
    edges.append(Link(l1, n2, 10000))
    edges.append(Link(l2, n1, 10000))
    edges.append(Link(l2, n2, 10000))
    edges.append(Link(l3, n3, 10000))
    edges.append(Link(l3, n4, 10000))
    edges.append(Link(l4, n3, 10000))
    edges.append(Link(l4, n4, 10000))
    edges.append(Link(l5, n5, 10000))
    edges.append(Link(l5, n6, 10000))
    edges.append(Link(l6, n5, 10000))
    edges.append(Link(l6, n6, 10000))
    edges.append(Link(l7, n7, 10000))
    edges.append(Link(l7, n8, 10000))
    edges.append(Link(l8, n7, 10000))
    edges.append(Link(l8, n8, 10000))
    edges.append(Link(l9, n9, 10000))
    edges.append(Link(l9, n10, 10000))
    edges.append(Link(l10, n9, 10000))
    edges.append(Link(l10, n10, 10000))
    edges.append(Link(l11, n11, 10000))
    edges.append(Link(l11, n12, 10000))
    edges.append(Link(l12, n11, 10000))
    edges.append(Link(l12, n12, 10000))
    edges.append(Link(l13, n13, 10000))
    edges.append(Link(l13, n14, 10000))
    edges.append(Link(l14, n13, 10000))
    edges.append(Link(l14, n14, 10000))
    edges.append(Link(l15, n15, 10000))
    edges.append(Link(l15, n16, 10000))
    edges.append(Link(l16, n15, 10000))
    edges.append(Link(l16, n16, 10000))
    # Inter leaf edges
    edges.append(Link(l1, l2, 40000))
    edges.append(Link(l3, l4, 40000))
    edges.append(Link(l5, l6, 40000))
    edges.append(Link(l7, l8, 40000))
    edges.append(Link(l9, l10, 40000))
    edges.append(Link(l11, l12, 40000))
    edges.append(Link(l13, l14, 40000))
    edges.append(Link(l5, l6, 40000))

    ################ Agg DC1 ###################
    g2 = Switch("Agg1Gateway")

    # Spine switches
    s9 = Switch("Agg1Spine1")
    s10 = Switch("Agg1Spine2")
    s11 = Switch("Agg1Spine3")
    s12 = Switch("Agg1Spine4")

    # Leaf switches
    l17 = Switch("Agg1Leaf1")
    l18 = Switch("Agg1Leaf2")
    l19 = Switch("Agg1Leaf3")   
    l20 = Switch("Agg1Leaf4")
    l21 = Switch("Agg1Leaf5")
    l22 = Switch("Agg1Leaf6")
    l23 = Switch("Agg1Leaf7")
    l24 = Switch("Agg1Leaf8")

    # Nodes
    n17 = Node("Agg1Node1", 60, 320, availability=0.9999)
    n18 = Node("Agg1Node2", 60, 320, availability=0.9999)
    n19 = Node("Agg1Node3", 60, 320, availability=0.9999)
    n20 = Node("Agg1Node4", 60, 320, availability=0.9999)
    n21 = Node("Agg1Node5", 60, 320, availability=0.9999)
    n22 = Node("Agg1Node6", 60, 320, availability=0.9999)
    n23 = Node("Agg1Node7", 60, 320, availability=0.9999)
    n24 = Node("Agg1Node8", 60, 320, availability=0.9999)

    locations += [g2, s9, s10, s11, s12, l17, l18, l19, l20, l21, l22, l23, l24,
                    n17, n18, n19, n20, n21, n22, n23, n24]

    # Gateway spines
    edges.append(Link(g2, s9, 100000))
    edges.append(Link(g2, s10, 100000))
    edges.append(Link(g2, s11, 100000))
    edges.append(Link(g2, s12, 100000))

    # Spines to Leafs
    edges.append(Link(s9, l17, 40000))
    edges.append(Link(s9, l18, 40000))
    edges.append(Link(s9, l19, 40000))
    edges.append(Link(s9, l20, 40000))
    edges.append(Link(s9, l21, 40000))
    edges.append(Link(s9, l22, 40000))
    edges.append(Link(s9, l23, 40000))
    edges.append(Link(s9, l23, 40000))
    edges.append(Link(s10, l17, 40000))
    edges.append(Link(s10, l18, 40000))
    edges.append(Link(s10, l19, 40000))
    edges.append(Link(s10, l20, 40000))
    edges.append(Link(s10, l21, 40000))
    edges.append(Link(s10, l22, 40000))
    edges.append(Link(s10, l23, 40000))
    edges.append(Link(s10, l24, 40000))
    edges.append(Link(s11, l17, 40000))
    edges.append(Link(s11, l18, 40000))
    edges.append(Link(s11, l19, 40000))
    edges.append(Link(s11, l20, 40000))
    edges.append(Link(s11, l21, 40000))
    edges.append(Link(s11, l22, 40000))
    edges.append(Link(s11, l23, 40000))
    edges.append(Link(s11, l24, 40000))
    edges.append(Link(s12, l17, 40000))
    edges.append(Link(s12, l18, 40000))
    edges.append(Link(s12, l19, 40000))
    edges.append(Link(s12, l20, 40000))
    edges.append(Link(s12, l21, 40000))
    edges.append(Link(s12, l22, 40000))
    edges.append(Link(s12, l23, 40000))
    edges.append(Link(s12, l24, 40000))
    # Leaf to node links
    edges.append(Link(l17, n17, 10000))
    edges.append(Link(l17, n18, 10000))
    edges.append(Link(l18, n17, 10000))
    edges.append(Link(l18, n18, 10000))
    edges.append(Link(l19, n19, 10000))
    edges.append(Link(l19, n20, 10000))
    edges.append(Link(l20, n19, 10000))
    edges.append(Link(l20, n20, 10000))
    edges.append(Link(l21, n21, 10000))
    edges.append(Link(l21, n22, 10000))
    edges.append(Link(l22, n21, 10000))
    edges.append(Link(l22, n22, 10000))
    edges.append(Link(l23, n23, 10000))
    edges.append(Link(l23, n24, 10000))
    edges.append(Link(l24, n23, 10000))
    edges.append(Link(l24, n24, 10000))
    # Inter leaf edges
    edges.append(Link(l17, l18, 40000))
    edges.append(Link(l19, l20, 40000))
    edges.append(Link(l21, l22, 40000))
    edges.append(Link(l23, l24, 40000))

    ################ Agg DC2 ###################
    g3 = Switch("Agg2Gateway")

    # Spine switches
    s13 = Switch("Agg2Spine1")
    s14 = Switch("Agg2Spine2")
    s15 = Switch("Agg2Spine3")
    s16 = Switch("Agg2Spine4")

    # Leaf switches
    l25 = Switch("Agg2Leaf1")
    l26 = Switch("Agg2Leaf2")
    l27 = Switch("Agg2Leaf3")   
    l28 = Switch("Agg2Leaf4")
    l29 = Switch("Agg2Leaf5")
    l30 = Switch("Agg2Leaf6")
    l31 = Switch("Agg2Leaf7")
    l32 = Switch("Agg2Leaf8")

    # Nodes
    n25 = Node("Agg2Node1", 60, 320, availability=0.9999)
    n26 = Node("Agg2Node2", 60, 320, availability=0.9999)
    n27 = Node("Agg2Node3", 60, 320, availability=0.9999)
    n28 = Node("Agg2Node4", 60, 320, availability=0.9999)
    n29 = Node("Agg2Node5", 60, 320, availability=0.9999)
    n30 = Node("Agg2Node6", 60, 320, availability=0.9999)
    n31 = Node("Agg2Node7", 60, 320, availability=0.9999)
    n32 = Node("Agg2Node8", 60, 320, availability=0.9999)

    locations += [g3, s13, s14, s15, s16, l25, l26, l27, l28, l29, l30, l31, l32,
                    n25, n26, n27, n28, n29, n30, n31, n32]

    # Gateway spines
    edges.append(Link(g3, s13, 100000))
    edges.append(Link(g3, s14, 100000))
    edges.append(Link(g3, s15, 100000))
    edges.append(Link(g3, s16, 100000))

    # Spines to Leafs
    edges.append(Link(s13, l25, 40000))
    edges.append(Link(s13, l26, 40000))
    edges.append(Link(s13, l27, 40000))
    edges.append(Link(s13, l28, 40000))
    edges.append(Link(s13, l29, 40000))
    edges.append(Link(s13, l30, 40000))
    edges.append(Link(s13, l31, 40000))
    edges.append(Link(s13, l32, 40000))
    edges.append(Link(s14, l25, 40000))
    edges.append(Link(s14, l26, 40000))
    edges.append(Link(s14, l27, 40000))
    edges.append(Link(s14, l28, 40000))
    edges.append(Link(s14, l29, 40000))
    edges.append(Link(s14, l30, 40000))
    edges.append(Link(s14, l31, 40000))
    edges.append(Link(s14, l32, 40000))
    edges.append(Link(s15, l25, 40000))
    edges.append(Link(s15, l26, 40000))
    edges.append(Link(s15, l27, 40000))
    edges.append(Link(s15, l28, 40000))
    edges.append(Link(s15, l29, 40000))
    edges.append(Link(s15, l30, 40000))
    edges.append(Link(s15, l31, 40000))
    edges.append(Link(s15, l32, 40000))
    edges.append(Link(s16, l25, 40000))
    edges.append(Link(s16, l26, 40000))
    edges.append(Link(s16, l27, 40000))
    edges.append(Link(s16, l28, 40000))
    edges.append(Link(s16, l29, 40000))
    edges.append(Link(s16, l30, 40000))
    edges.append(Link(s16, l31, 40000))
    edges.append(Link(s16, l32, 40000))
    # Leaf to node links
    edges.append(Link(l25, n25, 10000))
    edges.append(Link(l25, n26, 10000))
    edges.append(Link(l26, n25, 10000))
    edges.append(Link(l26, n26, 10000))
    edges.append(Link(l27, n27, 10000))
    edges.append(Link(l27, n28, 10000))
    edges.append(Link(l28, n17, 10000))
    edges.append(Link(l28, n28, 10000))
    edges.append(Link(l29, n29, 10000))
    edges.append(Link(l29, n30, 10000))
    edges.append(Link(l30, n29, 10000))
    edges.append(Link(l30, n30, 10000))
    edges.append(Link(l31, n31, 10000))
    edges.append(Link(l31, n32, 10000))
    edges.append(Link(l32, n31, 10000))
    edges.append(Link(l32, n32, 10000))
    # Inter leaf edges
    edges.append(Link(l25, l26, 40000))
    edges.append(Link(l27, l28, 40000))
    edges.append(Link(l29, l30, 40000))
    edges.append(Link(l31, l32, 40000))


    ################ Edge DC1 ###################
    g4 = Switch("Edge1Gateway")

    # Leaf switches
    l33 = Switch("Edge1Leaf1")
    l34 = Switch("Edge1Leaf2")
    l35 = Switch("Edge1Leaf3")   
    l36 = Switch("Edge1Leaf4")

    # Nodes
    n33 = Node("Edge1Node1", 60, 320, availability=0.9999)
    n34 = Node("Edge1Node2", 60, 320, availability=0.9999)
    n35 = Node("Edge1Node3", 60, 320, availability=0.9999)
    n36 = Node("Edge1Node4", 60, 320, availability=0.9999)

    locations += [g4, l33, l34, l35, l36, n33, n34, n35, n36]

    # Gateway spines
    edges.append(Link(g4, l33, 40000))
    edges.append(Link(g4, l34, 40000))
    edges.append(Link(g4, l35, 40000))
    edges.append(Link(g4, l36, 40000))
    # Leaf to node links
    edges.append(Link(l33, n33, 10000))
    edges.append(Link(l33, n34, 10000))
    edges.append(Link(l34, n34, 10000))
    edges.append(Link(l34, n34, 10000))
    edges.append(Link(l35, n35, 10000))
    edges.append(Link(l35, n36, 10000))
    edges.append(Link(l36, n35, 10000))
    edges.append(Link(l36, n36, 10000))
    # Inter leaf edges
    edges.append(Link(l33, l34, 40000))
    edges.append(Link(l35, l36, 40000))

    ################ Edge DC2 ###################
    g5 = Switch("Edge2Gateway")

    # Leaf switches
    l37 = Switch("Edge2Leaf1")
    l38 = Switch("Edge2Leaf2")
    l39 = Switch("Edge2Leaf3")   
    l40 = Switch("Edge2Leaf4")

    # Nodes
    n37 = Node("Edge2Node1", 60, 320, availability=0.9999)
    n38 = Node("Edge2Node2", 60, 320, availability=0.9999)
    n39 = Node("Edge2Node3", 60, 320, availability=0.9999)
    n40 = Node("Edge2Node4", 60, 320, availability=0.9999)

    locations += [g5, l37, l38, l39, l40, n37, n38, n39, n40]

    # Gateway spines
    edges.append(Link(g5, l37, 40000))
    edges.append(Link(g5, l38, 40000))
    edges.append(Link(g5, l39, 40000))
    edges.append(Link(g5, l40, 40000))
    # Leaf to node links
    edges.append(Link(l37, n37, 10000))
    edges.append(Link(l37, n38, 10000))
    edges.append(Link(l38, n37, 10000))
    edges.append(Link(l38, n38, 10000))
    edges.append(Link(l39, n39, 10000))
    edges.append(Link(l39, n40, 10000))
    edges.append(Link(l40, n39, 10000))
    edges.append(Link(l40, n40, 10000))
    # Inter leaf edges
    edges.append(Link(l37, l38, 40000))
    edges.append(Link(l39, l40, 40000))

    ################ Edge DC3 ###################
    g6 = Switch("Edge3Gateway")

    # Leaf switches
    l41 = Switch("Edge3Leaf1")
    l42 = Switch("Edge3Leaf2")
    l43 = Switch("Edge3Leaf3")   
    l44 = Switch("Edge3Leaf4")

    # Nodes
    n41 = Node("Edge3Node1", 60, 320, availability=0.9999)
    n42 = Node("Edge3Node2", 60, 320, availability=0.9999)
    n43 = Node("Edge3Node3", 60, 320, availability=0.9999)
    n44 = Node("Edge3Node4", 60, 320, availability=0.9999)

    locations += [g5, l41, l42, l43, l44, n41, n42, n43, n44]

    # Gateway spines
    edges.append(Link(g6, l41, 40000))
    edges.append(Link(g6, l42, 40000))
    edges.append(Link(g6, l43, 40000))
    edges.append(Link(g6, l44, 40000))
    # Leaf to node links
    edges.append(Link(l41, n41, 10000))
    edges.append(Link(l41, n42, 10000))
    edges.append(Link(l42, n41, 10000))
    edges.append(Link(l42, n42, 10000))
    edges.append(Link(l43, n43, 10000))
    edges.append(Link(l43, n44, 10000))
    edges.append(Link(l44, n43, 10000))
    edges.append(Link(l44, n44, 10000))
    # Inter leaf edges
    edges.append(Link(l41, l42, 40000))
    edges.append(Link(l43, l44, 40000))


    ################ Edge DC4 ###################
    g7 = Switch("Edge4Gateway")

    # Leaf switches
    l45 = Switch("Edge4Leaf1")
    l46 = Switch("Edge4Leaf2")
    l47 = Switch("Edge4Leaf3")   
    l48 = Switch("Edge4Leaf4")

    # Nodes
    n45 = Node("Edge4Node1", 60, 320, availability=0.9999)
    n46 = Node("Edge4Node2", 60, 320, availability=0.9999)
    n47 = Node("Edge4Node3", 60, 320, availability=0.9999)
    n48 = Node("Edge4Node4", 60, 320, availability=0.9999)

    locations += [g7, l45, l46, l47, l48, n45, n46, n47, n48]

    # Gateway spines
    edges.append(Link(g7, l45, 40000))
    edges.append(Link(g7, l46, 40000))
    edges.append(Link(g7, l47, 40000))
    edges.append(Link(g7, l48, 40000))
    # Leaf to node links
    edges.append(Link(l45, n45, 10000))
    edges.append(Link(l45, n46, 10000))
    edges.append(Link(l46, n45, 10000))
    edges.append(Link(l46, n46, 10000))
    edges.append(Link(l47, n47, 10000))
    edges.append(Link(l47, n48, 10000))
    edges.append(Link(l48, n47, 10000))
    edges.append(Link(l48, n48, 10000))
    # Inter leaf edges
    edges.append(Link(l45, l46, 40000))
    edges.append(Link(l47, l48, 40000))

    ################ Radio ###################
    # Radio Units
    r1 = Switch("Radio1")
    r2 = Switch("Radio2")
    r3 = Switch("Radio3")   
    r4 = Switch("Radio4")
    r5 = Switch("Radio5")
    r6 = Switch("Radio6")
    r7 = Switch("Radio7")   
    r8 = Switch("Radio8")
    r9 = Switch("Radio9")
    r10 = Switch("Radio10")
    r11 = Switch("Radio11")   
    r12 = Switch("Radio12")
    r13 = Switch("Radio13")
    r14 = Switch("Radio14")
    r15 = Switch("Radio15")   
    r16 = Switch("Radio16")
    r17 = Switch("Radio17")
    r18 = Switch("Radio18")
    r19 = Switch("Radio19")   
    r20 = Switch("Radio20")
    locations += [r1, r2, r3, r4, r5, r6, r7, r8, r9, r10,
                    r11, r12, r13, r14, r15, r16, r17, r18, r19, r20]
    
    ################ Transport ###################
    # Switches
    t1 = Switch("Transport1")
    t2 = Switch("Transport2")
    t3 = Switch("Transport3")   
    t4 = Switch("Transport4")
    t5 = Switch("Transport5")
    t6 = Switch("Transport6")
    t7 = Switch("Transport7")   

    locations += [t1, t2, t3, t4, t5, t6, t7]

    # Radio to transport edges.
    edges.append(Link(r1, t4, 10000, 5))
    edges.append(Link(r2, t4, 10000, 5))
    edges.append(Link(r3, t4, 10000, 5))
    edges.append(Link(r4, t4, 10000, 5))
    edges.append(Link(r5, t4, 10000, 5))
    edges.append(Link(r6, t4, 10000, 5))
    edges.append(Link(r7, t5, 10000, 5))
    edges.append(Link(r8, t5, 10000, 5))
    edges.append(Link(r9, t5, 10000, 5))
    edges.append(Link(r10, t5, 10000, 5))
    edges.append(Link(r11, t6, 10000, 5))
    edges.append(Link(r12, t6, 10000, 5))
    edges.append(Link(r13, t6, 10000, 5))
    edges.append(Link(r14, t6, 10000, 5))
    edges.append(Link(r15, t6, 10000, 5))
    edges.append(Link(r16, t7, 10000, 5))
    edges.append(Link(r17, t7, 10000, 5))
    edges.append(Link(r18, t7, 10000, 5))
    edges.append(Link(r19, t7, 10000, 5))
    edges.append(Link(r20, t7, 10000, 5))

    # Transport to Gateways
    edges.append(Link(t1, g1, 800000))
    edges.append(Link(t2, g2, 400000))
    edges.append(Link(t3, g3, 400000))
    edges.append(Link(t4, g4, 100000))
    edges.append(Link(t5, g5, 100000))
    edges.append(Link(t6, g6, 100000))
    edges.append(Link(t7, g7, 100000))

    # Transport to Transport
    edges.append(Link(t1, t2, 400000, 100))
    edges.append(Link(t1, t3, 400000, 100))
    edges.append(Link(t2, t4, 100000, 50))
    edges.append(Link(t2, t5, 100000, 50))
    edges.append(Link(t3, t6, 100000, 50))
    edges.append(Link(t3, t7, 100000, 50))


    network = Network("TestNetwork", locations, edges)
    network.save_as_json("data_used/networks/TestNetwork")

if __name__ == "__main__":
    main()