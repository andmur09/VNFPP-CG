# -*- coding: utf-8 -*-
from topology.network import Network
from topology.location import *
from topology.link import Link
from service_class.service import Service
from service_class.vnf import VNF
from optimisation.column_generation import ColumnGeneration

def main():
    # Makes the locations in the netork
    locations = [Switch("S1"), Switch("S2"), Switch("S3"), Switch("S4"), Switch("S5"), Node("N1", float(1), float(1)), Node("N2", float(1), float(1)), Node("N3", float(1), float(1))]

    # Makes the links between locations in the datacenter
    links = [Link(locations[0], locations[7], float(5), float(1)),
            Link(locations[7], locations[0], float(5), float(1)),
            Link(locations[0], locations[1], float(5), float(1)),
            Link(locations[1], locations[0], float(5), float(1)),
            Link(locations[0], locations[5], float(5), float(1)),
            Link(locations[5], locations[0], float(5), float(1)),
            Link(locations[1], locations[5], float(5), float(1)),
            Link(locations[5], locations[1], float(5), float(1)),
            Link(locations[1], locations[6], float(5), float(1)),
            Link(locations[6], locations[1], float(5), float(1)),
            Link(locations[2], locations[5], float(5), float(1)),
            Link(locations[5], locations[2], float(5), float(1)),
            Link(locations[2], locations[6], float(5), float(1)),
            Link(locations[6], locations[2], float(5), float(1)),
            Link(locations[5], locations[7], float(5), float(1)),
            Link(locations[7], locations[5], float(5), float(1)),
            Link(locations[4], locations[5], float(5), float(1)),
            Link(locations[5], locations[4], float(5), float(1)),
            Link(locations[6], locations[7], float(5), float(1)),
            Link(locations[7], locations[6], float(5), float(1)),
            Link(locations[3], locations[4], float(5), float(1)),
            Link(locations[4], locations[3], float(5), float(1)),
            Link(locations[3], locations[7], float(5), float(1)),
            Link(locations[7], locations[3], float(5), float(1)),
            Link(locations[3], locations[6], float(5), float(1)),
            Link(locations[6], locations[3], float(5), float(1)),
            Link(locations[4], locations[6], float(5), float(1)),
            Link(locations[6], locations[4], float(5), float(1))]

    # Creates then plots topology and saves to file
    problem = Network("trial1", locations, links)
    problem.save_as_dot()

    # Makes the services.
    functions = [VNF("Component0", 1, 1), VNF("Component1", 1, 1)]
    services = [Service("Service0", functions, 1, 10, 1, locations[0], locations[4])]

    # Initialises the optimisation problem.
    op = ColumnGeneration(problem, services)
    op.optimise()

    # Define source and sink for optimisation
    # _from = problem.getLocationByDescription("Gateway")
    # _to = problem.getLocationByDescription("Node1")
    # result = optimisation.minCostFlow(problem, stops)
    
    # # Defines source/sink pairs for each flow
    # gateway = problem.getLocationByDescription("Gateway")
    # node = problem.getLocationByDescription("Node1")
    # segments = [(gateway, node), (node, gateway)]

    # result = optimisation.minCostFlowWithStops(problem, segments)
    
if __name__ == "__main__":
    main()


