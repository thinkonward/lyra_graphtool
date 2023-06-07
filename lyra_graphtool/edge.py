from typing import List, TypeVar, Generic

from lyra_graphtool.vertex import Vertex

Edge = TypeVar('Edge')


# create and return edge object
class Edge(Generic[Edge]):
    def __init__(self, v1: Vertex, v2: Vertex):
        self.v1 = v1
        self.v2 = v2
        # use Linfinity distance
        # nearest-neighbors (d=1) is surrounding square
        self.length = max(abs(v1.x - v2.x), abs(v1.y - v2.y))

    # two edges are equal if they have the same vertices - order NOT important
    def __eq__(self, e: Edge) -> bool:
        vs1, vs2 = self.v1, self.v2
        ve1, ve2 = e.v1, e.v2
        eq = vs1 == ve1 and vs2 == ve2
        eq |= vs1 == ve2 and vs2 == ve1
        return eq

    # display info about edge
    def info(self, verbose=False) -> List:
        '''
        ## Displays list of information about the object
        
            Arguments:
                verbose : bool, default False

            Return: 
                list
        '''
        x0, y0 = self.v1.x, self.v1.y
        x1, y1 = self.v2.x, self.v2.y
        if verbose:
            print(f'[ ({x0},{y0}), ({x1},{y1}) ]')
        return [(x0, y0), (x1, y1)]

    # THIS DEFINES "nearest neighbor"
    # True - edge consists of "nearest neighbors", False otherwise
    def nearest_neighbor(self) -> bool:
        '''
        ## Defines "nearest neighbor"
        
            Arguments:
                None

            Return: 
                bool
                    True if edge consists of "nearest neighbors", False otherwise
        '''
        return self.length == 1.

    # True - edge is allowed for travel in 1 timestep: add to graph
    # False - cannot travel across edge in 1 timestep: do not add to graph
    def in_graph(self) -> bool:
        '''
        ## This determines if edge is allowed to travel in 1 timestep
        
            Arguments:
                None

            Return: 
                bool
                    True - edge is allowed for travel in 1 timestep: add to graph
                    False - cannot travel across edge in 1 timestep: do not add to graph
        '''
        return self.nearest_neighbor()