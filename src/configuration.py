import json
import pickle
from copy import deepcopy
from typing import Dict, Generic, List, Tuple, TypeVar

import numpy as np

from lyra_graphtool.edge import Edge
from lyra_graphtool.parameters import Parameters
from lyra_graphtool.worker import Worker, Worker_Type
from lyra_graphtool.vertex import Vertex, Vertex_Type


Config_Single_Time = TypeVar('Config_Single_Time')
Configuration = TypeVar('Configuration')


# single-time-step configuration
# Note: objects will have a deepcopy of the vertex. To modify a vertex
#       in the graph, use graph methods to get the graph's vertex.
class Config_Single_Time(Generic[Config_Single_Time]):

    def __init__(self, v: Vertex = None, access: bool = False):
        '''
        Class controls single-time-step configuration
        
            Note: objects will have a deepcopy of the vertex. To modify a vertex
                in the graph, use graph methods to get the graph's vertex.
        '''

        self.v = deepcopy(v)  # occupy this vertex at corresponding time
        self.access = access  # True: access this vertex's resources; False: don't access

    # display info about object
    def info(self, verbose=False) -> List:
        '''
        Displays list of information about the object
        
            Arguments:
                verbose : bool, default = False

            Return: 
                list
        '''
        if self.v is not None:
            x, y = self.v.x, self.v.y
            vtype = self.v.vertex_type
        else:
            x, y = None, None
            vtype = None
        access = self.access
        if verbose:
            print(f'[ ({x},{y}), {vtype}, acc={access} ]')
        return [(x, y), vtype, access]


# noinspection SpellCheckingInspection
class Configuration(Generic[Configuration]):

    def __init__(self, params: Parameters):
        '''
        Generic class for configuring budget, worker type, duration, graphing, and worker costs

            Arguments:
                params:
                    budget - 
                    
                    duration_time - 

                    graph - 

                    worker_cost_rate - 

            Return:
                None

        '''
        self.parameters = params
        self.budget = params.budget
        self.duration_time = params.duration_time
        self.graph = params.graph
        self.worker_cost_rate = params.worker_cost_rate

        # compute max number of workers of each type depending on budget
        budget = params.budget

        worker_types = list(Worker_Type.__members__.values())
        self.worker_types = worker_types

        max_workers = {}  # indexed by Worker_Type
        for wt in worker_types:
            max_workers[wt] = int(np.floor(budget / self.worker_cost_rate[wt]))

        self.max_workers = max_workers

        # Set up schedules over time for each worker
        #
        # A schedule is a time-series of pairs
        #       (Vertex, access_status) = Config_Single_Time object
        # for each worker
        #        access_status = True: then take resources at site = Vertex
        #                      = False: do not take resources, only travel through
        #
        # The time series runs from day 0 to day params.duration_days

        # access config by
        #    config[worker_type][worker number] = {0:config_single, 1:config_single,... }
        self.config = {}

        for worker_type in worker_types:

            self.config[worker_type] = {}
            for worker_number in range(max_workers[worker_type]):

                # set up path/schedule for single worker
                schedule = {}
                for t in range(self.duration_time):
                    config_t = Config_Single_Time()
                    schedule[t] = config_t

                self.config[worker_type][worker_number] = schedule

    # get vertices from which a worker's path must start
    #
    # Note: this allows for generalizations - should avoid hard-coding
    #       constraints outside this class (ie by solvers)
    def get_vertices_start(self) -> List:
        '''
        Get list the vertices from which a worker's path must start
            
                Note: this allows for generalizations - should avoid hard-coding
                    constraints outside this class (ie by solvers)
            
            Arguments:
                None
            
            Return:
                list
        '''

        verts_graph = self.graph.get_vertices_type(Vertex_Type.ORIGIN)

        verts_start = []
        for v in verts_graph:
            verts_start.append(deepcopy(v))

        return verts_start

    # add schedule (deepcopy) to configuration
    # at wt, wn
    def add_sched(self, wt: Worker_Type, wn: int, sched: Dict) -> None:
        '''    
        Add a schedule (deepcopy) to configuration at wt, wn
        
            Arguments:

                wt: Worker_Type (class), int

                wn: int, number of workers
                
                sched: dict, schedule
            
            Return:

                None
        '''

        if len(sched) == self.duration_time:
            self.config[wt][wn] = deepcopy(sched)
        else:
            raise ValueError(f'Schedule must have length {self.duration_time}.')

    # display info on given schedule
    @staticmethod
    def sched_info(sched: Dict) -> List:
        '''
        Display the info of the given schedule

            Arguments:

                sched: dict, schedule

            Return:
            
                list
        '''
        info = []
        for t in range(len(sched)):
            i = sched[t].info()  # [ (x,y), vtype, access ]
            info.append(f'[t={t}, ({i[0][0]},{i[0][1]}), vtype={i[1]}, acc={i[2]} ]')
        return info

    # Method to return Worker object, that has rates compliant with configuration
    def get_worker(self, wt: Worker_Type) -> Worker:
        '''
        Method to return Worker object, that has rates compliant with configuration

            Arguments:
                wt: int, Worker_Type(class)\n

            Return:
                Worker(class)
        '''
        return Worker(wt, rates=self.worker_cost_rate)

    # Method returning dict of workers used in configuration
    def get_current_workers(self, config: Dict) -> Dict:
        '''
        Method returning dict of workers used in configuration

            Arguments:
                config: dict

            Returns: 
                dict
        '''
        used_workers = {}
        for wt in self.worker_types:
            used_workers[wt] = 0
            for wn, sched in config[wt].items():
                not_na = [1 if cs.v is not None else 0 for _, cs in sched.items()]
                if np.sum(not_na) > 0:
                    used_workers[wt] += 1
        return used_workers

    # Method to return sites accessed in schedule
    # Return: dict with key (x,y) , value access_count: int
    # Note: access_count = 1 means the site v appeared v.time_to_acquire adjacent times in a schedule
    def get_accessed_sites(self, schedule: Dict) -> Tuple[Dict, str]:
        '''
        Method to return sites accessed in a schedule
        
                Note: access_count = 1 means the site v appeared v.time_to_acquire adjacent times in a schedule

            Parameters:
                schedule: dict
            
            Return: 
                dict with key (x,y) 

                value access_count: int
        '''
        accesses = {(vertex.x, vertex.y): 0 for vertex in self.graph.vertices
                    if vertex.vertex_type in self.graph.vertices[0].accessible_types()
                    }
        worker_message = "Log of accesses:\n"

        for wt in self.worker_types:
            for wn, sched in schedule[wt].items():
                t = 0
                while t < len(sched):
                    curr_sched = sched[t]
                    if curr_sched.access is True:
                        t = t + max(curr_sched.v.time_to_acquire - 1, 0)
                        accesses[(curr_sched.v.x, curr_sched.v.y)] += 1
                        worker_message += f"{wt.__str__()} number {wn} accessed " \
                                          f"{curr_sched.v.vertex_type.__str__()} at " \
                                          f"({curr_sched.v.x}, {curr_sched.v.y}) at time step {t}\n"
                    t += 1
        return accesses, worker_message

    # Method to return maximum revenue for graph
    def get_max_revenue(self):
        '''
        Method to return maximum revenue for graph

            Arguments:
                None

            Return:
                int, maximum revenue
                    max_revenue = amount_of_sites * single_reward
        '''
        max_revenue = 0
        for site_type in self.graph.vertices[0].accessible_types():
            amount_of_sites = len(self.graph.get_vertices_type(site_type))
            if amount_of_sites < 1:
                continue
            single_reward = self.graph.get_vertices_type(site_type)[0].reward
            max_revenue += amount_of_sites * single_reward
        return max_revenue

    # True if site v is being accessed at time t in configuration
    # False otherwise
    def site_accessed_at_time(self, v: Vertex, t: int) -> bool:
        '''
        Returns bool value to determine if site is being accessed at time t in configuration

            Arguments:
                v: Vertex object, from lyra_graphtool.vertex
                
                t: int, time step value 

            Returns: 
                bool 
                    True if site v is being accessed at time t in configuration, False otherwise
        '''

        for wt in self.worker_types:
            for wn in range(len(self.config[wt])):
                max_time = len(self.config[wt][wn]) - 1

                if t > max_time:
                    continue
                else:
                    cs = self.config[wt][wn][t]
                    if cs.v is not None:
                        if cs.v == v and cs.access:
                            return True

        return False

    # True if site v is being accessed in configuration
    # False otherwise
    def site_accessed(self, v: Vertex) -> bool:
        '''
        Returns bool value to determine if site is being accessed in configuration

            Arguments:
                v: Vertex object, from lyra_graphtool.vertex

            Returns: 
                bool 
                    True if site v is being accessed in configuration, False otherwise
        '''
        for wt in self.worker_types:
            for wn in range(len(self.config[wt])):
                for t in range(len(self.config[wt][wn])):
                    cs = self.config[wt][wn][t]
                    if cs.v is not None:
                        if cs.v == v and cs.access:
                            return True
        return False

    # get length of a one worker schedule, how many timesteps was it active?
    @staticmethod
    def get_sched_path_length(sched: Dict) -> int:
        '''
        Method to return length of a one worker schedule. Used to determine how many timesteps was the worker active.

            Arguments:
                sched: dict, schedule 

            Returns: 
                int, number of active timesteps
        '''
        length = 0
        for t, cs in sched.items():
            if cs.v is not None:
                length += 1
        return length

    # True if schedule for single worker is empty
    # False if otherwise
    @staticmethod
    def is_empty(sched: Dict) -> bool:
        '''
        Method to determine if single worker scedule is empty

            Arguments:
                sched: dict, schedule 

            Returns: 
                bool
                    True if schedule for single worker is empty, False if otherwise
        '''
        for t, cs in sched.items():
            if cs.v is not None:
                return False
        return True

    # Test if schedule is feasible spatially
    #   Note: a schedule is a sequence of Config_Single_Time objects
    #
    # 1. must start at ORIGIN type vertex if origin_flag==True
    # 2. ending vertex followed by None vertices
    # 3. if hire worker again at later time, must start at ORIGIN
    #
    # If origin_flag==False, schedule can start/restart at any vertex
    #
    # sched is Dict with values Config_Single_Time
    #
    def sched_feasible_space(self, sched: Dict, origin_flag=True) -> bool:
        '''
        Method to determine if schedule is spatially feasible

                Criteria
                    1. Must start at ORIGIN type vertex if origin_flag==True
                    2. Ending vertex followed by None vertices
                    3. If hire worker again at later time, must start at ORIGIN
                    Note: If origin_flag==False, schedule can start/restart at any vertex

            Arguments:
                sched: dict, schedule
                    Note: a schedule is a sequence of Config_Single_Time objects
                    
                origin_flag: bool, default True

            Returns: 
                bool
                    True if schedule is spatially feasible, false if otherwise
        '''
        if len(sched) != self.duration_time:
            raise ValueError(f'Schedule must have {self.duration_time} keys to match duration_time.')

        if sched[0].v is not None and origin_flag:  # must start at origin if specified
            if sched[0].v.vertex_type != Vertex_Type.ORIGIN:
                return False

        # make sure adjacent vertices are in edge set
        v_prev = sched[0].v  # this may be None
        for t in range(1, self.duration_time):

            if sched[t].v is None:  # current vertex is None
                v_prev = None
                continue

            else:  # actual vertex:  path[t].v is not None

                if v_prev is None:  # vertex following None
                    if origin_flag:
                        if sched[t].v.vertex_type != Vertex_Type.ORIGIN:
                            return False
                        else:
                            v_prev = sched[t].v
                            continue  # ORIGIN following None is ok

                    else:  # any vertex can follow None
                        v_prev = sched[t].v
                        continue

                elif sched[t].v == v_prev:  # can stay at previous location
                    v_prev = sched[t].v
                    continue

                else:  # different location: vertex follows vertex (both not None)

                    # must be an edge in graph
                    e = Edge(v_prev, sched[t].v)
                    if e not in self.graph.edges:
                        return False
                    else:  # it is an edge
                        v_prev = sched[t].v
        return True

    #
    # Test all schedules for spatial feasibility
    #
    def sched_all_feasible_space(self, origin_flag=True) -> bool:
        '''
        Method to determine if ALL schedules are spatially feasible

            Arguments:
                origin_flag: bool, default True

            Returns: 
                bool
                    True if schedule is spatially feasible, false if otherwise
        '''

        for worker_type in self.worker_types:

            for worker_number in range(len(self.config[worker_type])):

                sched = self.config[worker_type][worker_number]

                feas = self.sched_feasible_space(sched, origin_flag)

                if not feas:
                    return False
        return True

    # Test if a schedule's access/extract properties are feasible
    #
    # 1. Can only access site - cannot access a non-site vertex as BASIC, ORIGIN
    # 2. Access constraint - must have Worker >= SITE
    # 3. If accessing, must access site for v.days_to_acquire
    # 4. Cannot access after expiration time
    # 5. Cannot access a site more than once
    #
    def sched_feasible_access_sites(self, sched: Dict, worker_type: Worker_Type) -> bool:
        '''
        Method to determine if schedule's access/extract properties are feasible

                Criteria
                    1. Can only access site - cannot access a non-site vertex as BASIC, ORIGIN
                    2. Access constraint - must have Worker >= SITE
                    3. If accessing, must access site for v.days_to_acquire
                    4. Cannot access after expiration time
                    5. Cannot access a site more than once

            Arguments:
                sched: dict, schedule
                    Note: a schedule is a sequence of Config_Single_Time objects
                    
                worker_type: Worker_Type (class), from lyra_graphtool.worker

            Returns: 
                bool
                    True if schedule's access/extract properties are feasible, False if otherwise
        '''

        sites_accessed = []
        t = 0
        while t < len(sched):

            v = sched[t].v
            acc = sched[t].access

            if v is None:
                if acc is not False:
                    return False
                else:
                    t += 1
                    continue

            if v.vertex_type == Vertex_Type.BASIC or v.vertex_type ==  Vertex_Type.ORIGIN:
                if acc is True:
                    return False
                else:
                    t += 1

            else:  # default - this vertex is a site
                if acc is True:

                    # test if sites was previously accessed
                    if v in sites_accessed:
                        return False
                    else:
                        sites_accessed.append(v)

                    # test if worker allowed to access
                    if self.get_worker(worker_type).access(v) is False:
                        return False

                    # test if accessing after expire
                    if v.expiration_time is not None:
                        if t >= v.expiration_time:
                            return False

                    # set future times to same vertex with access=True
                    for t_off in range(1, v.time_to_acquire):
                        if t + t_off < self.duration_time:
                            s2 = sched[t + t_off]
                            v2, acc2 = s2.v, s2.access
                        else:  # ran out of time: infeasible
                            return False

                        if v2 is None:
                            return False
                        elif v2 != v or acc2 is False:
                            return False

                    t += v.time_to_acquire
                else:
                    t += 1

        return True

    #
    # Test all schedules for access feasibility
    #
    # 1. Only one worker can access a specific site at a single point in time
    # 2. Once a site has been accessed, it can no longer be accessed at subsequent times
    #
    # noinspection SpellCheckingInspection
    def sched_all_feasible_access_sites(self) -> bool:
        '''
        Method to determine if schedule's access properties are feasible

                Criteria
                    1. Only one worker can access a specific site at a single point in time
                    2. Once a site has been accessed, it can no longer be accessed at subsequent times

            Arguments:
                None

            Return: 
                bool
                    True if schedule's access properties are feasible, False if otherwise
        '''

        # check schedule feasibility for each individual worker
        for worker_type in self.worker_types:

            for worker_number in range(len(self.config[worker_type])):

                sched = self.config[worker_type][worker_number]

                feas = self.sched_feasible_access_sites(sched, worker_type)

                if not feas:
                    return False

        # check that each site is accessed no more than one time
        accesses, _ = self.get_accessed_sites(self.config)

        if max(accesses.values()) > 1:
            return False

        return True

    #
    # Calculate cost of a single-worker schedule
    #
    # sched is a dict with key=time, value=Config_Single_Time
    #   (for a single worker)
    #
    @staticmethod
    def cost_sched(sched: Dict, worker: Worker) -> float:
        '''
        Calculate the cost of a single-worker schedule

            Arguments:
                sched: dict,
                    sched is a dict with key=time, value=Config_Single_Time (for a single worker)
                    worker: Worker (class), from lyra_graphtool.worker

            Returns:
                float, cost value of a single worker schedule
        '''

        cost_sched = 0
        for t in range(len(sched)):
            v = sched[t].v
            access = sched[t].access

            if v is None:  # worker not on payroll
                continue

            else:  # worker is on payroll

                cost_base = worker.worker_cost_rate
                cost = cost_base

                if access is True:

                    # get multipliers
                    mult_time = v.mult_time
                    mult_time_start = v.mult_time_active[0]
                    mult_time_end = v.mult_time_active[1]
                    mult_worker = v.mult_worker

                    # cost for time multiplier
                    if mult_time_start is not None and mult_time_end is not None:
                        if mult_time_start <= t <= mult_time_end:
                            cost += (mult_time - 1) * cost_base

                    # cost for site multiplier
                    wt = worker.worker_type
                    if wt in mult_worker.keys():
                        cost += (mult_worker[wt] - 1) * cost_base

                cost_sched += cost

        return cost_sched

    #
    # Calculate cost of entire configuration
    #
    # sched is a dict with key=time, value=Config_Single_Time
    #   (for a single worker)
    #
    # NOTE: default worker_cost_rate is used for each worker type
    def cost(self) -> float:
        '''
        Calculate the cost of the entire configuration
                Note: default worker_cost_rate is used for each worker type

            Arguments:
                None

            Returns:
                float, cost value of entire configuration
        '''

        cost_config = 0

        for wt in self.worker_types:
            for wn in range(len(self.config[wt])):
                sched = self.config[wt][wn]
                worker = self.get_worker(wt)
                cost_config += self.cost_sched(sched, worker)

        return cost_config

    # Budget Feasibility of Configuration
    def budget_feasible(self) -> bool:
        '''
        Method to determine if budget is feasible for configuration. If Cost <= Budget then Feasible else Not Feasible

            Arguments:
                None

            Returns:
                bool
                    True if budget is feasible, False if otherwise
        '''

        if self.cost() <= self.budget:
            return True
        else:
            return False

    # test overall feasibility of config using above methods
    def feasible(self) -> bool:
        '''
        Method to test the overall feasibily using sched_all_feasibility_space()
        , sched_all_feasible_access_sites(), and budget_feasible()

            Arguments:
                None

            Returns:
                bool
                    True if all functions are feasible, False if otherwise
        '''

        feas_space = self.sched_all_feasible_space()
        feas_access = self.sched_all_feasible_access_sites()
        feas_budget = self.budget_feasible()

        feas = feas_space and feas_access and feas_budget

        return feas

    # calculate returns/revenue from schedule for single worker
    #
    # sched should be tested for feasibility by sched_feasible_access_sites
    # before calling this funciton
    #
    # Note: revenue depends on worker_type
    @staticmethod
    def sched_revenue(sched: Dict):
        '''
        Method to calculate revenue from schedule for single worker
                Note: Sched should be tested for feasibility by sched_feasible_access_sites
                before calling this function

            Arguments:
                sched: dict, schedule
                    sched is a dict with key=time, value=Config_Single_Time (for a single worker)
                    worker: Worker (class), from lyra_graphtool.worker

            Returns:
                int, revenue (depends on worker_type)
        '''

        revenue = 0
        t = 0
        while t < len(sched):

            v = sched[t].v
            acc = sched[t].access

            if acc is False:  # this covers non-SITEx vertices and .v = None
                t += 1
                continue

            # at SITEx vertex AND accessing
            revenue += v.reward
            t += v.time_to_acquire

        return revenue

    # calculate returns/revenue for entire configuration
    #
    # sched should be tested for feasibility by sched_feasible_access_sites
    # before calling this funciton
    #
    def revenue(self):
        '''
        Method to calculate revenue from schedule for entire configuration
                Note: Sched should be tested for feasibility by sched_feasible_access_sites
                before calling this function

            Arguments:
                None

            Returns:
                int, revenue value for entire configuration
        '''

        revenue = 0

        for wt in self.worker_types:
            for wn in range(len(self.config[wt])):
                sched = self.config[wt][wn]
                revenue += self.sched_revenue(sched)

        return revenue

    # save configuration to file
    def save(self, file_name: str) -> None:
        '''
        Save configuration file to pickle format

            Arguments:
                filename: str, filepath for file to be saved

            Returns:
                None
        '''
        fn = file_name

        with open(fn, 'wb') as f:
            pickle.dump(self, f)

    # save Configuration.config schedule dictionary to json
    def save_to_json(self, file_name: str) -> None:
        '''
        Save configuration schedule dictionary to json format

            Arguments:
                filename: str, filepath for file to be saved

            Returns:
                None
        '''
        config_json = {}
        for wt in self.worker_types:
            config_json[wt] = {}
            for wn in range(len(self.config[wt])):
                config_json[wt][wn] = {}
                sched = self.config[wt][wn]
                for t in range(len(sched)):
                    cs = sched[t]
                    v, access = cs.v, cs.access
                    if v is None:
                        cs_basic = ['None', cs.access]
                    else:
                        cs_basic = [float(v.x), float(v.y), cs.access]
                    config_json[wt][wn][t] = cs_basic
        # save to json
        with open(file_name, 'w') as f:
            json.dump(config_json, f)

    # load json config into Configuration.config
    def load_from_json(self, file_name: str) -> None:
        '''
        Load configuration schedule dictionary from json format into Configuration.config

            Arguments:
                filename: str, json filepath to import

            Returns:
                None
        '''

        def jsonKeys2int(x):
            if isinstance(x, dict):
                return {int(k): v for k, v in x.items()}
            return x

        with open(file_name) as f:
            d = json.load(f, object_hook=jsonKeys2int)

        config = {}
        for wt in self.worker_types:
            config[wt] = {}
            for wn in range(len(self.config[wt])):
                config[wt][wn] = {}
                sched = self.config[wt][wn]
                for t in range(len(sched)):
                    if wt in d.keys():
                        if wn in d[wt].keys():
                            if t in d[wt][wn].keys():
                                csj = d[wt][wn][t]
                                if csj[0] == 'None':
                                    config[wt][wn][t] = Config_Single_Time()
                                else:
                                    msg_err = f'json dictionary in file {file_name} must have triple (float,float,bool) ' + \
                                              f'at index {[wt, wn, t]}.'
                                    if len(csj) != 3:
                                        raise ValueError(msg_err)
                                    x, y, access = csj[0], csj[1], csj[2]
                                    if not isinstance(x, float) or not isinstance(y, float) or not isinstance(access,
                                                                                                              bool):
                                        raise ValueError(msg_err)
                                    v = self.graph.get_vertex_xy(x, y)
                                    if v is None:
                                        raise ValueError(
                                            f'Vertex in json file {file_name} with coords {(x, y)} not found in graph.')

                                    config[wt][wn][t] = Config_Single_Time(v, access)

        self.config = config  # no errors, copy config

    # override python deepcopy function, which is very slow
    # 1. use pickle.dumps() to write object to binary file
    # 2. use pickle.loads() to read binary file into new object
    # Argument
    #        memo: dummy variable to match format of deepcopy from copy library
    # Returns
    #        Node: deepcopy of original configuration
    #
    def __deepcopy__(self, memo):
        return pickle.loads(pickle.dumps(self, -1))  # -1 uses most recent protocol

