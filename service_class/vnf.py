import itertools
import copy
from multiprocessing import cpu_count

class VNF(object):
    id_iter = itertools.count()
    """
    Class representing a VNF used in a service
        \param description      String description of component
        \param cpu              Float required CPU
        \param ram              Float required RAM
    """
    def __init__(self, description: str, cpu: float, ram: float):
        self.id = next(VNF.id_iter)
        self.description = description
        self.cpu = cpu
        self.ram = ram
    
    def __str__(self):
        """
        string representation of link
        """
        return "Component: {}, Description: {}".format(self.id, self.description)

    def copy(self):
        """
        returns a copy of the component
        """
        return VNF(self.description[:], self.cpu, self.ram)
