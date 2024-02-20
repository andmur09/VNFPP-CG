import json
import sys
import csv
import os

if __name__ == "__main__":
    """
    This script takes a solution json and appends results to csv file.
    """
    # command line arguments
    if len(sys.argv) != 3:
        raise ValueError("Script should take 2 arguments:\n\t 1. The path to the result file.\n\t 2. The path to the csv file.")
    results = []
    file_path = sys.argv[1]
    csv_path = sys.argv[2]
    tokens = file_path.split("/")
    file_name = tokens[-1]

    with open(file_path) as f:
        data = json.load(f)

    # Gets experiment params and results
    result = {}
    result["network"] = data["network"]["name"]
    result["nservices"] = file_name.split("_")[1][9:]
    result["loadfactor"] = file_name.split("_")[2][10:]
    result["nodesused"] = data["number of nodes used"]
    result["lpobj"] = data["lower bound"]
    result["obj"] = data["upper bound"]
    result["totpens"] = data["total penalties"]
    result["avpens"] = data["availability penalties"]
    result["tppens"] = data["throughput penalties"]
    result["ltpens"] = data["latency penalties"]
    result["time"] = data["runtime"]


    logfile = "/home/kpb20194/DCopt/gurobi_files/{}".format(file_name[:-4] + "log")
    with open(logfile) as f:
        for line in f:
            if "Explored" in line:
                tokens = line.split(" ")
                index_seconds = tokens.index("seconds")
                mip_runtime = float(tokens[index_seconds - 1])
    result["miptime"] = mip_runtime
    result["cgtime"] = data["runtime"] - mip_runtime
    results.append(result)

    keys = results[0].keys()
    if os.path.exists(csv_path):
        # Tries to open with write - this should raise value error if file doesnt exist.
        with open(csv_path, "a", newline='') as f:
            writer = csv.DictWriter(f, keys)
            writer.writerows(results)
    else:
        vals = result.values()
        with open(csv_path, "w+", newline='') as f:
            writer = csv.DictWriter(f, keys)
            writer.writeheader()
            writer.writerows(results)

