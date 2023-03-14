import itertools
import re

inf = 1e9

class Location(object):
    id_iter = itertools.count()
    """
    represents a location in the datacenter.
    """
    def __init__(self, description: str, type = None):
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
    def __init__(self, description: str):
        super().__init__(description)
        self.type = "Switch"


class Node(Location):
    """
    Represents a node location.
    cpu: CPU available on the node.
    ram: RAM available on the node.
    availability: Availability of node, MTTF/(MTTR+MTTF)
    cost: Node rental cost.
    active: Whether the node is active (False to simulate node failure).
    """
    def __init__(self, description: str, cpu: int, ram: float, cost: float = float(1), availability: float = float(1), active: bool = True):
        super().__init__(description)
        self.type = "Node"
        self.cpu = cpu
        self.ram = ram
        self.cost = cost
        self.availability = availability
        self.active = active

    def get_component_assigned(self) -> str:
        """
        If node is a node in a service graph representing the assignment of a component to a node
        this will return the component description else it will return None
        """
        if self.type == "Node":
            s = re.search(r"\[(\w+)\]", self.description)
            if s != None:
                return s.group(0)[1:-1]
        return None

    def to_json(self) -> dict:
        """
        return a dictionary for use with json.
        """
        return {"id": self.id, "description": self.description, "type": self.type, "cpu": self.cpu, "ram": self.ram, "cost": self.cost,"availability": self.availability}

class Dummy(Node):
    """
    Represents a dummy node location. We use this to initialise the column generation. The dummy node is an artificial node with arbitrarily
    high CPU and RAM such that every function can be hosted here. However it has arbitrarily high cost to if another placement is possible,
    the optimisation will chose the different placement. As a result any service with a function placed on the dummy node can be conidered
    as failed.
    """
    def __init__(self, description):
        super().__init__(description, int(inf), inf, cost = inf)
        