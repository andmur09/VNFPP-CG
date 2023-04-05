import itertools
import re

inf = 1e9

class Location(object):
    id_iter = itertools.count()
    """
    represents a location in the datacenter.
    """
    def __init__(self, description: str = None, type = None):
        self.id = next(Location.id_iter)
        self.description = description
        self.type = type
    
    def __str__(self) -> str:
        """
        prints string representation of location
        """
        return "Location: {}, Description: {}".format(self.id, self.description)
    
    def copy(self):
        """
        returns a copy of the location
        """
        return Location(self.description[:], type = self.type)

    def to_json(self) -> dict:
        """
        return a dictionary for use with json.
        """
        return {"id": self.id, "description": self.description, "type": self.type}

class Switch(Location):
    """
    Represents a switch location.
    """
    def __init__(self, description: str = None):
        super().__init__(description)
        self.type = "Switch"

    def load_from_dict(self, dictionary):
        """
        Given a json dictionary, loads the attributes.
        """
        assert list(dictionary.keys()) == ["id", "description", "type"], "Keys in JSON don't match expected input for type switch."
        self.id, self.description, self.type = dictionary["id"], dictionary["description"], dictionary["type"]

    def copy(self):
        """
        returns a copy of the location
        """
        return Switch(self.description[:])

class Node(Location):
    """
    Represents a node location.
    cpu: CPU available on the node.
    ram: RAM available on the node.
    availability: Availability of node, MTTF/(MTTR+MTTF)
    cost: Node rental cost.
    active: Whether the node is active (False to simulate node failure).
    """
    def __init__(self, description: str = None, cpu: int = 1, ram: float = float(1), cost: float = float(1), availability: float = float(1), active: bool = True):
        super().__init__(description)
        self.type = "Node"
        self.cpu = cpu
        self.ram = ram
        self.cost = cost
        self.availability = availability
        self.active = active

    def to_json(self) -> dict:
        """
        return a dictionary for use with json.
        """
        return {"id": self.id, "description": self.description, "type": self.type, "cpu": self.cpu, "ram": self.ram, "cost": self.cost,"availability": self.availability}
    
    def load_from_dict(self, dictionary):
        """
        Given a json dictionary, loads the attributes.
        """
        self.id, self.description, self.type, self.cpu, self.ram, self.cost, self.availability = dictionary["id"], dictionary["description"], dictionary["type"], dictionary["cpu"], dictionary["ram"], dictionary["cost"], dictionary["availability"]

    def copy(self):
        """
        returns a copy of the location
        """
        return Node(self.description[:], cpu = self.cpu, ram = self.ram, cost = self.cost, availability=self.availability, active=self.active)

class Dummy(Node):
    """
    Represents a dummy node location. We use this to initialise the column generation. The dummy node is an artificial node with arbitrarily
    high CPU and RAM such that every function can be hosted here. However it has arbitrarily high cost to if another placement is possible,
    the optimisation will chose the different placement. As a result any service with a function placed on the dummy node can be conidered
    as failed.
    """
    def __init__(self, description):
        super().__init__(description, int(inf), inf, cost = inf)
    
    def copy(self):
        """
        returns a copy of the location
        """
        return Dummy(self.description)
        