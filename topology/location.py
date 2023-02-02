import itertools
import re

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
    

class Gateway(Location):
    """
    Represents a datacenter gateway node.
    """
    def __init__(self, description: str):
        super().__init__(description)
        self.type = "Gateway"

class SuperSpine(Location):
    """
    Represents a super spine switch location.
    """
    def __init__(self, description: str):
        super().__init__(description)
        self.type = "SuperSpine"

class Spine(Location):
    """
    Represents a spine switch location.
    """
    def __init__(self, description: str):
        super().__init__(description)
        self.type = "Spine"

class Leaf(Location):
    """
    Represents a leaf switch location.
    """
    def __init__(self, description: str):
        super().__init__(description)
        self.type = "Leaf"


class Node(Location):
    """
    Represents a node switch location.
    cpu: CPU available on the node.
    ram: RAM available on the node.
    cost: Node rental cost.
    active: Whether the node is active (False to simulate node failure).
    """
    def __init__(self, description: str, cpu: float, ram: float, cost: float = float(1), active: bool = True):
        super().__init__(description)
        self.type = "Node"
        self.cpu = cpu
        self.ram = ram
        self.cost = cost
        self.active = active

    def deactivate(self):
        if self.active == True:
            self.active = False
    
    def activate(self):
        if self.active == False:
            self.active = True

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
        return {"id": self.id, "description": self.description, "type": self.type, "ram": self.ram, "cpu": self.cpu, "cost": self.cost}

