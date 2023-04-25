from cgi import test
from http import server
import numpy as np
import os
from service_class.vnf import VNF
from service_class.service import Service
from topology.network import Network
from optimisation.column_generation import ColumnGeneration
import random
import numpy as np

def solve_case(topology, vnfs, service_requests, filename, verbose = 0):
    """
    Solves a case using given load and services and saves result to .json
    """
    #Solves minimising SLA violations.
    try:
        cg = ColumnGeneration(topology, vnfs, service_requests, weights = [1, 0], verbose = verbose)
        cg.optimise()
        cg.parse_solution()
        cg.save_as_json(filename = "data_used/results/" + filename + "_cg")
    except:
       pass

    # # Solves using heuristic.
    try:
        heur = ColumnGeneration(topology, vnfs, service_requests, weights = [1, 0], verbose = verbose)
        heur.optimise(use_heuristic=True, max_iterations = 0)
        heur.parse_solution()
        heur.save_as_json(filename = "data_used/results/" + filename + "_greedy")
    except:
        pass

if __name__ == "__main__":
    """
    This script runs the experiments used in the results.
    """
    # Loads the network
    network_file = "data_used/networks/TestNetwork2"
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
    service_names = [s.description for s in services]

    # Increases number of service requests
    for n_requests in range(10, 101, 10):
        print("N requests: ", n_requests)
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
        
        fname = network.description + "_nservices{}".format(n_requests)
        solve_case(network, vnfs, service_requests, fname, verbose = 0)