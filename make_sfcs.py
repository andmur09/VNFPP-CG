import os
from service_class.vnf import VNF
from service_class.service import Service
from topology.network import Network
import numpy as np
import random
from itertools import product
import json

def random_sfcs(network, services, n_requests, scale_factors = [1]):
    """
    Given a number of requests, samples service types and returns list of SFC requests.

    Params
    -----------------------------
        network:        topology.network.Network
                            instance of Network to use.
        services:       list
                            list of services to use.
        n_requests:     int
                            number of SFC requests.
        scale_factor:   list[int]
                            number of users per service request (i.e. if 2 then total required throughput is 2 * service.throughput)
    Returns
    -----------------------------
                        list
    """
    service_names = [s.description for s in services]
    n_subscribers = {s.description: 0 for s in services}
    service_probs = [s.percentage_traffic/100 for s in services]
    service_probs = [s/sum(service_probs) for s in service_probs]
    assert sum(service_probs) == 1, "Sum of probabilities for services must be one. Please modify service.percentage_traffic."

    # Randomly assigns service requests to service types.
    for i in range(n_requests):
        service_type = np.random.choice(service_names, p = service_probs)
        n_subscribers[service_type] += 1

    access_points = [l for l in network.locations if l.handles_requests == True]
    sfcs = [[] for i in range(len(scale_factors))]

    for service in services:
        # This keeps track of which source/sink combinations have been used so that service chains are unique
        combinations = list(product(access_points, access_points))
        # Randomly samples source/sink pairs for requests.
        samples = random.sample(combinations, n_subscribers[service.description])
        count = 1
        for sample in samples:
            source, sink = sample[0], sample[1]
            # Adds service requests.
            for i in range(len(scale_factors)):
                sfcs[i].append(Service(service.description + str(count), service.vnfs[:], scale_factors[i] * service.throughput,
                service.latency, service.percentage_traffic, service.availability, source = source, sink = sink, n_subscribers = scale_factors[i], weight = service.weight))
                count += 1
    return sfcs

if __name__ == "__main__":
    """
    This script generates the SFCs used in the experiments and saves them in data_used/sfcs/
    """

    network_file = "data_used/networks/abilene"
    vnfs_dir = "data_used/vnfs/"
    service_dir = "data_used/slices/"
    output_dir = "data_used/sfcs/"

    # Loads the network
    network = Network()
    network.load_from_json(network_file)

    vnfs = []
    # Loads the set of vnfs.                                                                                              
    for file in os.listdir(vnfs_dir):
        function = VNF()
        function.load_from_json(vnfs_dir + file)
        vnfs.append(function)
    
    services = []
    # Loads the set of slices.
    for file in os.listdir(service_dir):
        service = Service()
        service.load_from_json(service_dir + file)
        services.append(service)

    scale_factors = [1]

    for n in [3]:
        sfcs = random_sfcs(network, services, n, scale_factors = scale_factors)
        for i in range(len(scale_factors)):
            sfc_dict = {}
            name = network.description + "_nservices{}_loadfactor{}".format(n, i + 1)
            sfc_dict["name"] = name
            sfc_dict["network"] = network.description
            sfc_dict["sfcs"] = [s.to_json() for s in sfcs[i]]
            with open(output_dir + name + ".json", 'w') as fp:
                json.dump(sfc_dict, fp, indent=4, separators=(", ", ": "))
