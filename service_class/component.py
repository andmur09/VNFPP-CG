import itertools
import copy

class Component(object):
    id_iter = itertools.count()
    """
    Class representing a component used in a service
        \param description      String description of component
        \param requirements     Dictionary of component requirements. At the moment this is of the form {"cpu": float, "ram": float}
        \param replica_count    Integer defining how many replicas of a component are required.
    """
    def __init__(self, description: str, requirements: dict, replica_count: int):
        self.id = next(Component.id_iter)
        self.description = description
        self.requirements = requirements
        self.replica_count = replica_count
    
    def __str__(self):
        """
        string representation of link
        """
        return "Component: {}, Description: {}".format(self.id, self.description)

    def copy(self):
        """
        returns a copy of the component
        """
        return Component(self.description[:], copy.deepcopy(self.requirements), self.replica_count)
