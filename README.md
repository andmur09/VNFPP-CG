# VNFPP-CG (VNF Placement Problem - Column Generation)

This library consists of a set of tools for solving the VNF placement problem using column generation.

In Network Function Virtualisation (NFV), network functions are virtualised and hosted on industrial standard high volume servers. A request to route traffic through a sequence of Virtual Network Functions (VNF) is denoted a Service Function Chain (SFC). The VNF placement problem involves computing the optimal placement of VNF's and provisioning of SFC's within the physical network infrastructure, subject to hardware restrictions and the Quality of Service (QoS) constraints outlined in the Service Level Agreement (SLA). In 5G and beyond, the QoS constraints are expected to become more tailored to cope with diverse service use cases. In this paper, we present a VNF placement algorithm based on the column generation method, which computes the optimal number and placement of VNF's and routing of SFC's while maximising QoS. SLA constraints (latency, data-rate and availability) are modelled as soft constraints for which violation incurs a cost which is then minimised. We validate our approach against a heuristic greedy algorithm on a multi-tiered Radio Access Network (RAN) and show that the column generation method offers a significant reduction in SLA violation cost at the expense of increased run-time. We also highlight that satisfying QoS (in particular availability) can dramatically increase the number of host nodes required, thus a trade-off exists between SLA violation cost and operational cost which should be explored further.

Current modules include:

- topology
    - network - A class to model telecoms networks, where the network is composed of a graph made up of a ser of links and locations.
    - location - A class to model physical locations within the network. These can be:
	    - Switch - A location solely used for routing.
		- Node - A location with compute and memory resources that can be used to host VNF's.
	- link - A class to model physical links between two locations in the network.
	
- service_class
	- vnf - A class used to model virtual network functions.
	- service - A class used to model a service request. Each service request has a flow that must be processed by an ordered sequence of service requests.
	- graph - A class used to model a augmented service graph. This is used as the "shortest path" column generation sub prolem.
	- path - A class used to model a path on the augmented service graph.

- optimisation
    - column_generation - A class used to model VNF-PP as a column generation optimisation problem. This iterates between two problems, the Restricted Master Problem (rmp) and the column generation problem (pricing_problem).

### Requirements

VNFPP-CG makes use of type hinting generics (e.g. `l : list[str] = ()`) introduced in **Python 3.9**.

Install using git:

```bash

git clone

```

VNFPP-CG has the following dependencies:

- [https://www.gurobi.com/](GUROBI) (Linear programming solver used in the optimisation. A free academic license is available on their website.
- [https://graphviz.org/](Graphviz) (Used to plot topology.)

### Example

To run the experiments used in this paper run the script toy_example.py. This loads the RAN test topology, set of VNF's and service types. It randomly samples 10 service
requests and then solves using both column generation and a greedy heuristic. Set verbose = 0 for no output, verbose = 1 for logfile, verbose = 2 for logfile .lp and .ilp
output. The result will be saved as a .json file, output directory can be changed by modifying fname.