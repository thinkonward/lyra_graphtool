import random
from copy import deepcopy
from random import randint

import lyra_graphtool as lgtool
from lyra_graphtool import Configuration, Config_Single_Time, Edge, Graph, Graph_Type, Parameters, Vertex, Worker_Type, Vertex_Type

def random_alg(co_orig:lgtool.Configuration, access_probability:int = 0.5) -> int:
    '''
    Randomly assigns workers to sites and schedules them to access sites randomly.
    Returns profit and Configuration object.
    '''
    
    # create config with empty scedules; ie all Config_Single_Time = (None, False)
    co = deepcopy(co_orig)
    worker_types = co.worker_types
    for wt in worker_types:
        for wn in range(len(co.config[wt])):
            for t in range(len(co.config[wt][wn])):
                co.config[wt][wn][t] = lgtool.Config_Single_Time()

    current_workers = co.get_current_workers(co.config)

    available_workers = 0
    for wt in worker_types:
        available_workers += co.max_workers[wt]

    # start all schedules at ORIGIN
    v_orig = co.graph.get_vertices_type(lgtool.Vertex_Type.ORIGIN)[0]
    cs_orig = lgtool.Config_Single_Time(v = v_orig)

    # add workers until exceed budget or no more workers available
    cost_config = co.cost()
    cost_under_budget = True
    while cost_under_budget and available_workers > 0:

        # TYPE: choose a random worker type
        enough_workers = False
        while not enough_workers:
            wt_rand = randint(0,len(worker_types)-1)
            # can we add one more worker of chosen type?
            enough_workers = current_workers[wt_rand] < co.max_workers[wt_rand]

        wt = wt_rand                   # worker type
        wn = current_workers[wt_rand]  # worker number

        sched = deepcopy(co.config[wt][wn])   # schedule to create

        w = co.get_worker(wt)

        # DURATION
        # get random duration and random start time
        duration = randint(1, co.duration_time - 1)
        time_start = randint(0, co.duration_time - duration - 1)
        time_end = time_start + duration

        # start at origin
        # add to schedule and update cost
        sched_or = deepcopy(sched)
        sched_or[time_start] = cs_orig
        co.config[wt][wn] = sched_or
        cost_config = co.cost()

        if cost_config > co.budget:
            cost_under_budget = False
            co.config[wt][wn] = sched    # set sched back to original
            continue

        # random walk
        cs_prev = cs_orig
        t = time_start + 1
        while t < time_end:

            sched = deepcopy(co.config[wt][wn])  # remember current sched

            # pick random next vertex
            verts_adj = co.graph.adjacent_vertices(cs_prev.v)
            ind_next_vert = randint(0, len(verts_adj)-1)
            v_next = verts_adj[ind_next_vert]
            # ACCESS
            if w.access(v_next) and not co.site_accessed_at_time(v_next,t):
                p = random.uniform(0,1)
                access = True if p <= access_probability else False
            else:
                access = False

            # set next Config_Single_Time's in schedule if accessing
            time_to_acq = v_next.time_to_acquire
            config_t_set = False     # True when a config is found for this case
            if access:
                if t + time_to_acq < co.duration_time:
                    # enough time to access

                    # set up schedule that accesses site
                    cs_access = lgtool.Config_Single_Time(v_next, access=True)
                    sched_access = deepcopy(sched)
                    for ta in range(t, t + time_to_acq):
                        sched_access[ta] = deepcopy(cs_access)

                    # set config schedule to access: is this within budget AND not being accessed?
                    co.config[wt][wn] = sched_access
                    feas = co.feasible()

                    if feas:
                        config_t_set = True
                        t += v_next.time_to_acquire    # move forward after access
                        cs_prev = deepcopy(cs_access)

                    else:  
                        co.config[wt][wn] = sched      # set back to original undecided sched

            if config_t_set == False:  # either not accessing, or not enough time to access
                # try no access config
                cs_no_access = lgtool.Config_Single_Time(v_next, access=False)
                sched_no_access = deepcopy(sched)
                sched_no_access[t] = deepcopy(cs_no_access)

                co.config[wt][wn] = sched_no_access

                within_budget = co.cost() < co.budget

                if within_budget:
                    config_t_set = True
                    t += 1                         # move forward after access
                    cs_prev = deepcopy(cs_no_access)
                else:  
                    co.config[wt][wn] = sched      # set back to original undecided sched
            if config_t_set == False:   # over budget in every case
                cost_under_budget = False    # break out of outside while over workers
                break                        # break out of time while

        current_workers = co.get_current_workers(co.config) # update count of current workers
        available_workers -= 1
        
    if co.cost() > 0:
        profit = co.revenue() - co.cost()
    else:
        profit = None
    
    return profit, co

