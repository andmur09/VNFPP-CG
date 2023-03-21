import itertools
import json
inf = 1e9

class VNF(object):
    id_iter = itertools.count()
    """
    Class representing a VNF used in a service
        \param description      String description of vnf
        \param cpu              Float required CPU (number of cores)
        \param ram              Float required RAM (Mb)
        \param throughput       Float throughput traffic processing capabilities per instance of VNF (Mbps)
        \param latency          Float delay caused by VNF (ms)
        \param availability     Float availability of VNF
    """
    def __init__(self, description: str = None, cpu: float = 1, ram: float = 1, throughput: float = None, latency: float = 0, availability: float = 1):
        self.id = next(VNF.id_iter)
        self.description = description
        self.cpu = cpu
        self.ram = ram
        self.throughput = throughput
        self.latency = latency
        self.availability = availability
        self.cuts_generated = 0
    
    def __str__(self):
        """
        string representation of link
        """
        return self.description

    def copy(self):
        """
        returns a copy of the component
        """
        return VNF(self.cpu, self.ram, self.throughput, self.latency, self.availability)
    
    def to_json(self) -> dict:
        """
        Returns a json dictionary describing the vnf.
        """
        return {"name": self.description, "cpu": self.cpu, "ram": self.ram, "throughput": self.throughput, "latency": self.latency, "availability": self.availability}

    def save_as_json(self, filename = None):
        """
        saves the network as a JSON to filename.json
        """
        to_dump = self.to_json()
        if filename != None:
            if filename[-5:] != ".json":
                filename = filename + ".json"
        else:
            filename = self.description + ".json"

        with open(filename, 'w') as fp:
            json.dump(to_dump, fp, indent=4, separators=(", ", ": "))
    
    def load_from_json(self, filename: str):
        """
        Given a json file, it loads the data and stores as instance of VNF.
        """
        if filename[-5:] != ".json":
            filename = filename + ".json"

        # Opens the json and extracts the data
        with open(filename) as f:
            data = json.load(f)
        assert list(data.keys()) == ["name", "cpu", "ram", "throughput", "latency", "availability"], "Keys in JSON don't match expected input for type vnf."
        self.description, self.cpu, self.ram, self.throughput, self.latency, self.availability = data["name"], data["cpu"], data["ram"], data["throughput"], data["latency"], data["availability"]