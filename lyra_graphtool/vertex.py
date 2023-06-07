from enum import IntEnum
from typing import List, TypeVar, Generic

import lyra_graphtool.utils as utils

Vertex = TypeVar('Vertex')


class Vertex_Type(IntEnum):
    '''
    ## Used to define type of location.
        Note: BASIC is a location in space, neither a SITE nor ORIGIN
    '''
    BASIC = 0   # location in space; neither a site nor origin
    ORIGIN = 1
    SITE1 = 2
    SITE2 = 3
    SITE3 = 4
    OTHER = 5
    OTHER2 = 6


# create and return a vertex object
class Vertex(Generic[Vertex]):

    def __init__(self, x, y,
                 v_type=Vertex_Type.BASIC,
                 reward=0,  # reward given
                 mult_time=1,  # multiplied by cost
                 mult_time_active=(None, None),  # mult_time applies t1 <= time <= t2 (t1,t2)
                 mult_worker=None,  # multiplied by cost for this type, eg {Worker_Type.WORKER1: 1.5}
                 expiration_time=None,  # reward = 0 after this time (in days)
                 time_to_acquire=0,  # site must be accessed for this many time steps
                 ):
        '''
        ## Used to create and return a vertex object

            Arguments:

                v_type: Vertex_Type.BASIC

                reward: int, default = 0
                    reward given

                mult_time: int, default = 1
                    multiplied by cost

                mult_time_active: int pair, default = (None, None)
                    mult_time applies t1 <= time <= t2 (t1,t2)

                mult_worker: int, default = None
                    multiplied by cost for this type, eg {Worker_Type.WORKER1: 1.5}

                expiration_time: int, default = None
                    reward = 0 after this time (in days)

                time_to_acquire: int, default = 0
                    site must be accessed for this many time steps

            Return:
                None
        '''
        if mult_worker is None:
            mult_worker = {}
        self.x = x
        self.y = y
        self.vertex_type = v_type
        self.reward = reward
        self.mult_time = mult_time
        self.mult_time_active = mult_time_active
        self.mult_worker = mult_worker
        self.expiration_time = expiration_time
        self.time_to_acquire = time_to_acquire

    def __eq__(self, v: Vertex) -> bool:
        eq = self.x == v.x and self.y == v.y
        return eq

    # display info about vertex
    def info(self, verbose=False) -> List:
        '''
        ## Get information about vertex

            Arguments:
                verbose: bool, default=False

            Return:
                list, vertices information list
        '''
        x, y = self.x, self.y
        v_type = self.vertex_type
        if verbose:
            print(f'[ ({x},{y}), {v_type} ]')
        return [(x, y), v_type]

    @staticmethod
    def accessible_types() -> List:
        return [Vertex_Type.SITE1, Vertex_Type.SITE2, Vertex_Type.SITE3]

