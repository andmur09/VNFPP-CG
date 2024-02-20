# VNFPRP-CG (VNF Placement and Routing Problem - Column Generation)

This library consists of a set of tools for solving the VNF placement problem using column generation.

In Network Function Virtualisation (NFV), network functions are virtualised and hosted on industrial standard high volume servers. A request to route traffic through a sequence of Virtual Network Functions (VNF) is denoted a Service Function Chain (SFC). The VNF Placement and Routing Problem (VNF-PRP) involves computing the optimal placement of VNFs and provisioning of SFCs within the physical network infrastructure, subject to hardware restrictions and the Quality of Service (QoS) constraints outlined in the Service Level Agreement (SLA). In 5G and beyond, the QoS constraints are expected to become more tailored to cope with diverse service use cases. In this repository, we present a VNF placement and routing algorithm based on the column generation method, which computes the optimal number and placement of VNFs and routing of SFCs while maximising QoS. SLA constraints (latency, data-rate and availability) are modelled as soft constraints for which violation incurs a cost which is then minimised.

Current modules include:

- topology
    - network - A class to model telecoms networks, where the network is composed of a graph made up of a number of links and locations.
    - location - A class to model physical locations within the network. These can be:
	    - Switch - A location solely used for routing.
		- Node - A location with compute and memory resources that can be used to host VNFs.
	- link - A class to model physical links between two locations in the network.
	
- service_class
	- vnf - A class used to model VNFs.
	- service - A class used to SFCs. Each service request has a flow that must be processed by an ordered sequence of VNFs.
	- graph - A class used to model an augmented service graph. This is used as the "shortest path" column generation sub prolem.
	- path - A class used to model a path on the augmented service graph.

- optimisation
    - column_generation - A class used to model VNF-PRP as a column generation optimisation problem. This iterates between two problems, the Restricted Master Problem (rmp) and the Column Generation Problem (pricing_problem).

### Requirements

VNFPP-CG makes use of type hinting generics (e.g. `l : list[str] = ()`) introduced in **Python 3.9**.

Install using git:

```bash

git clone

```

VNFPP-CG has the following dependencies:

- [https://www.gurobi.com/](GUROBI) (Linear programming solver used in the optimisation. A free academic license is available on their website.)
- [https://graphviz.org/](Graphviz) (Used to plot topology.)

### Results
Results are contained in the folder data_used/results and can be updating the cases list in run_experiments.py and running the script.