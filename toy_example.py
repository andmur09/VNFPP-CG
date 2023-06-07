from cgi import test
from http import server
import numpy as np
import os
from service_class.vnf import VNF
from service_class.service import Service
from topology.location import Switch, Node
from topology.network import Network
from optimisation.column_generation import ColumnGeneration
from optimisation.compact_model import CompactModel
import random
import numpy as np

def solve_case(topology, vnfs, service_requests, filename, verbose = 0):
    """
    Solves a case using given load and services and saves result to .json
    """
    
    compact = CompactModel(topology, vnfs, service_requests, weights = [100, 1], verbose = verbose)
    compact.optimise()
    compact.parse_solution()
    compact.save_as_json(filename = filename + "_mip")

    # #Solves minimising SLA violations.
    # try:
    #     cg = ColumnGeneration(topology, vnfs, service_requests, weights = [1, 0], verbose = verbose)
    #     cg.optimise()
    #     cg.parse_solution()
    #     cg.save_as_json(filename = filename + "_cg")
    # except:
    #    pass

    # # # Solves using heuristic.
    # try:
    #     heur = ColumnGeneration(topology, vnfs, service_requests, weights = [1, 0], verbose = verbose)
    #     heur.optimise(use_heuristic=True, max_iterations = 0)
    #     heur.parse_solution()
    #     heur.save_as_json(filename = filename + "_greedy")
    # except:
    #     pass

def randomise_service_ran(network, services, n_requests):
    """
    Given a number of service requests it randomly samples service types and radio station source nodes. Returns a set of n_requests service requests.
    """
    service_names = [s.description for s in services]
    n_subscribers = {s.description: 0 for s in services}
    # Randomly assigns service requests to service types.
    for i in range(n_requests):
        subscriber = random.choice(service_names)
        n_subscribers[subscriber] += 1
    service_requests = []
    # Using number of subscribers from each service, assigns them to radio towers.
    for service in services:
        # Gets rid of services that have no subscribers.
        if n_subscribers[service.description] != 0:
            # Gets list of radio stations from which service requests can originate
            radio_stations = [l for l in network.locations if "Radio" in l.description]
            chunks = [0 for tower in radio_stations]
            for n in range(n_subscribers[service.description]):
                to_assign = random.randint(0, len(radio_stations) - 1)
                chunks[to_assign] += 1
            # Aggregates the number of subscribers across radio tower sources.
            for t in range(1, len(radio_stations) + 1):
                if chunks[t-1] != 0:
                    service_requests.append(Service(service.description + str(t), service.vnfs[:], chunks[t-1] * service.throughput,
                    service.latency, service.percentage_traffic, service.availability, radio_stations[t-1], n_subscribers = chunks[t-1]))
    return service_requests

def randomise_service_requests(network, services, n_requests):
    """
    Given a number of service requests it randomly samples service types and radio station source nodes. Returns a set of n_requests service requests.
    """
    service_names = [s.description for s in services]
    n_subscribers = {s.description: 0 for s in services}
    # Randomly assigns service requests to service types.
    for i in range(n_requests):
        subscriber = random.choice(service_names)
        n_subscribers[subscriber] += 1
    service_requests = []
    # Using number of subscribers from each service, assigns them to radio towers.
    for service in services:
        # Gets rid of services that have no subscribers.
        if n_subscribers[service.description] != 0:
            # Gets 
            switches = [l for l in network.locations if not isinstance(l, Node)]
            chunks = [[0 for l in switches] for l in switches]
            for n in range(n_subscribers[service.description]):
                source, sink = 0, 0
                while source == sink:
                    source = random.randint(0, len(switches) - 1)
                    sink = random.randint(0, len(switches) - 1)
                chunks[source][sink] += 1                
            # Aggregates the number of subscribers across radio tower sources.
            count = 0
            for i in range(len(switches)):
                for j in range(len(switches)):
                    if chunks[i][j] != 0:
                        service_requests.append(Service(service.description + str(count), service.vnfs[:], chunks[i][j] * service.throughput,
                        service.latency, service.percentage_traffic, service.availability, source = switches[i], sink = switches[j], n_subscribers = chunks[i][j]))
                        count += 1
    return service_requests


if __name__ == "__main__":
    """
    This script runs the experiments used in the results.
    """
    fname = "test"
    # Loads the network
    network_file = "data_used/networks/abilene"
    network = Network()
    network.load_from_json(network_file)

    vnfs = []
    # Loads the set of vnfs.                                                                                              
    for file in os.listdir("data_used/vnfs/"):
        function = VNF()
        function.load_from_json("data_used/vnfs/" + file)
        vnfs.append(function)
    
    services = []
    # Loads the set of services.
    for file in os.listdir("data_used/sfcs/"):
        service = Service()
        service.load_from_json("data_used/sfcs/" + file)
        services.append(service)
    
    requests = randomise_service_requests(network, services, 10)
    solve_case(network, vnfs, requests, fname, verbose = 2)