import os
from service_class.vnf import VNF
from service_class.service import Service
from topology.network import Network
from optimisation.column_generation import ColumnGeneration
import numpy as np
import json

def solve_case(topology, vnfs, service_requests, results_dir, filename, verbose = 0):
    """
    Solves a case using given load and services and saves result to .json
    """
    #Solves minimising SLA violations.
    cg = ColumnGeneration(topology, vnfs, service_requests, weights = [1, 0], verbose = verbose, log_dir= "gurobi_files/", name = filename + "_heur")
    cg.optimise()
    cg.parse_solution()
    cg.save_as_json(filename = results_dir + filename + "_heur")

def load_sfcs(sfc_path, network):
    """
    This script takes a path to a file containing json sfcs and loads it and returns a list of sfcs.
    """
    sfcs = []
    with open(sfc_path) as f:
        data = json.load(f)
        for s in data["sfcs"]:
            # Finds the source and the sink node in the provided network instance.
            source, sink = None, None
            for location in network.locations:
                if location.description == s["sink"]["description"] and location.description == s["source"]["description"]:
                    source = location
                    sink = location
                if location.description == s["sink"]["description"]:
                    sink = location
                elif location.description == s["source"]["description"]:
                    source = location
            assert source != None and sink != None, "Source and Sink not found in current graph."
            if s["latency"] != None and s["availability"] != None:
                sfc = Service(s["name"], s["vnfs"], throughput = float(s["throughput"]),
                        latency = float(s["latency"]), availability = float(s["availability"]), source = source, sink = sink,
                        n_subscribers = int(s["n_subscribers"]), weight = float(s["weight"]))
            elif s["latency"] != None and s["availability"] == None:
                sfc = Service(s["name"], s["vnfs"], throughput = float(s["throughput"]),
                        latency = float(s["latency"]), source = source, sink = sink,
                        n_subscribers = int(s["n_subscribers"]), weight = float(s["weight"]))
            else:
                sfc = Service(s["name"], s["vnfs"], throughput = float(s["throughput"]),
                        source = source, sink = sink,
                        n_subscribers = int(s["n_subscribers"]), weight = float(s["weight"]))
            sfcs.append(sfc)
    return sfcs


if __name__ == "__main__":
    """
    This script runs the experiments used in the results.
    """

    network_file = "data_used/networks/nobeleu"
    vnfs_dir = "data_used/vnfs/"
    sfcs_dir = "data_used/sfcs/"
    results_dir = "data_used/results_1it/"
    cases = ["nobeleu_nservices700_loadfactor5.json"]
    print("Loading network")
    
    # Loads the network
    network = Network()
    network.load_from_json(network_file)
    vnfs = []
    # Loads the set of vnfs.                                                                                              
    for file in os.listdir(vnfs_dir):
        function = VNF()
        function.load_from_json(vnfs_dir + file)
        vnfs.append(function)
    
    for case in cases:
        # Loads sfcs
        sfcs = load_sfcs(sfcs_dir + case, network)
        fname = case.split(".")[0]
        print("solving case {}".format(case))
        solve_case(network, vnfs, sfcs, results_dir, fname, verbose = 2)
