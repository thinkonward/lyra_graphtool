from enum import IntEnum
from functools import total_ordering
from typing import Dict, Generic, TypeVar

from lyra_graphtool.vertex import Vertex, Vertex_Type

Worker = TypeVar('Worker')


class Worker_Type(IntEnum):
    WORKER1 = 0
    WORKER2 = 1
    WORKER3 = 2


WORKER_COST_RATE = {
        Worker_Type.WORKER1: 0,
        Worker_Type.WORKER2: 0,
        Worker_Type.WORKER3: 0
    }


@total_ordering
class Worker(Generic[Worker]):

    def __init__(self, w_type: Worker_Type = Worker_Type.WORKER1, rates: Dict = None):

        self.worker_type = w_type
        self.worker_cost_rate = rates[w_type] or WORKER_COST_RATE[w_type]

    # worker == site ?
    def __eq__(self, obj) -> bool:
        if type(obj) == Vertex:
            w1 = Worker_Type.WORKER1
            w2 = Worker_Type.WORKER2
            w3 = Worker_Type.WORKER3
            s1 = Vertex_Type.SITE1
            s2 = Vertex_Type.SITE2
            s3 = Vertex_Type.SITE3
            e = self.worker_type == w1 and obj.vertex_type == s1
            e |= self.worker_type == w2 and obj.vertex_type == s2
            e |= self.worker_type == w3 and obj.vertex_type == s3
            return e
        elif type(obj) == Worker:
            e = self.worker_type == obj.worker_type
            return e
        else:
            raise ValueError(f"Can't compare Worker to type {str(type(obj))}")

    # worker >= site ?
    def __ge__(self, obj) -> bool:
        w1 = Worker_Type.WORKER1
        w2 = Worker_Type.WORKER2
        w3 = Worker_Type.WORKER3
        if type(obj) == Vertex:
            s1 = Vertex_Type.SITE1
            s2 = Vertex_Type.SITE2
            s3 = Vertex_Type.SITE3
            ge = self.worker_type == w3 and obj.vertex_type in [s1, s2, s3]
            ge |= self.worker_type == w2 and obj.vertex_type in [s1, s2]
            ge |= self.worker_type == w1 and obj.vertex_type == s1
            return ge
        elif type(obj) == Worker:
            ge = self.worker_type == w3 and obj.worker_type in [w1, w2, w3]
            ge |= self.worker_type == w2 and obj.worker_type in [w1, w2]
            ge |= self.worker_type == w1 and obj.worker_type == w1
            return ge
        else:
            raise ValueError(f"Can't compare Worker to type {str(type(obj))}")

    # can worker access site ?
    def access(self, vertex: Vertex) -> bool:
        s1 = Vertex_Type.SITE1
        s2 = Vertex_Type.SITE2
        s3 = Vertex_Type.SITE3
        sites = [s1, s2, s3]

        if vertex.vertex_type in sites:
            return self >= vertex

        else:  # vertex is not a site: origin or basic
            return False

