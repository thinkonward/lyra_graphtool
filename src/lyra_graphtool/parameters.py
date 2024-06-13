import datetime
import itertools
import json
import os
import pickle
from argparse import ArgumentParser
from copy import deepcopy
from typing import Generic, List, TypeVar

import lyra_graphtool.utils as utils
from lyra_graphtool.graph import Graph, Graph_Type
from lyra_graphtool.worker import WORKER_COST_RATE, Worker_Type
from lyra_graphtool.vertex import Vertex_Type


Parameters = TypeVar('Parameters')


class Parameters(Generic[Parameters]):
    def __init__(self, graph: Graph, budget=utils.BUDGET, duration_time=utils.DURATION,
                 cost_rate=WORKER_COST_RATE):
        self.budget = budget
        self.duration_time = duration_time
        self.graph = graph
        self.worker_cost_rate = cost_rate

    def display(self):
        print(vars(self))


# noinspection DuplicatedCode,SpellCheckingInspection
class ProcessArgs:
    """
    Set up args and graph. Get graph from this class object after instantiating.
    """

    def __init__(self, args_list: List = None, verbose=False):
        # noinspection PyUnresolvedReferences
        """
                set up args
                Arguments:
                    args_list - list of form

                                    ['--budget', '1000', '--duration', '20', ...]

                                args followed by values.
                                If args_list = '', default values will be used.

                        Note: If args_list contains a value for 'filename_graph',
                                the graph will be loaded (pickle file).
                                specified args will OVERRIDE graph parameters.
                              If there is no value for 'filename_graph', a graph will be created
                              using arg parameters.

                        verbose: bool, default = False

                Returns:
                    None
                """

        # set these below
        self.args = None
        self.args_trial = None
        self.graph = None
        self.site_structures = None
        self.worker_cost_rate = None

        self.keys_gen = [
            'trial_name', 'filename_graph',
            # graph geometric properties
            'max_x', 'max_y', 'num_verts', 'graph_type',
            'num_site1', 'num_site2', 'num_site3',
            # worker properties
            'worker1_cost', 'worker2_cost', 'worker3_cost',
            # configuration params
            'budget', 'duration'
        ]

        self.keys_site = [  # site economics properties
            'site1_acquire_time', 'site2_acquire_time', 'site3_acquire_time',
            'site1_reward', 'site2_reward', 'site3_reward',
            'site1_mult_time', 'site2_mult_time', 'site3_mult_time',
            'site1_mult_time_active', 'site2_mult_time_active', 'site3_mult_time_active',
            'site1_mult_workers', 'site2_mult_workers', 'site3_mult_workers',
            'site1_exp_time', 'site2_exp_time', 'site3_exp_time'
        ]

        parser = ArgumentParser(description='Args for Lyra')

        # trial name
        parser.add_argument('--trial_name', type=str, required=False,
                            default="Lyra_trial", help='name of trial')

        # graph
        parser.add_argument('--filename_graph', type=str, required=False, default="",
                            help='filename of graph - overloads arg values')
        parser.add_argument('--max_x', type=utils.pint, required=False, default=utils.NotSpecI,
                            help='maximum x value of graph vertex coordinates (0<=x<=max_x)')
        parser.add_argument('--max_y', type=utils.pint, required=False, default=utils.NotSpecI,
                            help='maximum y value of graph vertex coordinates (0<=y<=max_y)')
        parser.add_argument('--num_verts', type=utils.pint, required=False, default=utils.NotSpecI,
                            help='number of vertices in graph')
        parser.add_argument('--graph_type', type=utils.grstr, required=False, default="",
                            help='type of graph: "random" or "grid"')
        parser.add_argument('--num_site1', type=utils.nnint, required=False, default=utils.NotSpecI,
                            help='number of graph vertices of site-type 1')
        parser.add_argument('--num_site2', type=utils.nnint, required=False, default=utils.NotSpecI,
                            help='number of graph vertices of site-type 2')
        parser.add_argument('--num_site3', type=utils.nnint, required=False, default=utils.NotSpecI,
                            help='number of graph vertices of site-type 3')

        # site properties
        parser.add_argument('--site1_acquire_time', type=utils.pint, required=False, default=utils.NotSpecI,
                            help='time steps to aquire reward from site-type 1')
        parser.add_argument('--site2_acquire_time', type=utils.pint, required=False, default=utils.NotSpecI,
                            help='time steps to aquire reward from site-type 2')
        parser.add_argument('--site3_acquire_time', type=utils.pint, required=False, default=utils.NotSpecI,
                            help='time steps to aquire reward from site-type 3')

        parser.add_argument('--site1_reward', type=utils.nnfloat, required=False, default=utils.NotSpecF,
                            help='reward amount from site-type 1')
        parser.add_argument('--site2_reward', type=utils.nnfloat, required=False, default=utils.NotSpecF,
                            help='reward amount from site-type 2')
        parser.add_argument('--site3_reward', type=utils.nnfloat, required=False, default=utils.NotSpecF,
                            help='reward amount from site-type 3')

        parser.add_argument('--site1_mult_time', type=utils.nnfloat, required=False, default=utils.NotSpecF,
                            help='time multiplier value for all site-type 1')
        parser.add_argument('--site2_mult_time', type=utils.nnfloat, required=False, default=utils.NotSpecF,
                            help='time multiplier value for all site-type 2')
        parser.add_argument('--site3_mult_time', type=utils.nnfloat, required=False, default=utils.NotSpecF,
                            help='time multiplier value for all site-type 3')

        parser.add_argument('--site1_mult_time_active', type=utils.comma_separated_int_2, required=False,
                            default=utils.NotSpecI, help='time multiplier is active between these times for all '
                            'site-type 1 (if only one arg t, active time >= t)')
        parser.add_argument('--site2_mult_time_active', type=utils.comma_separated_int_2, required=False,
                            default=utils.NotSpecI, help='time multiplier is active between these times for all '
                            'site-type 2 (if only one arg t, active time >= t)')
        parser.add_argument('--site3_mult_time_active', type=utils.comma_separated_int_2, required=False,
                            default=utils.NotSpecI, help='time multiplier is active between these times for all '
                            'site-type 3 (if only one arg t, active time >= t)')

        parser.add_argument('--site1_mult_workers', type=utils.comma_separated_float_3, required=False,
                            default=utils.NotSpecF,
                            help='site-type 1 worker multipliers for w1, w2, w3 (missing args set to 1)')
        parser.add_argument('--site2_mult_workers', type=utils.comma_separated_float_3, required=False,
                            default=utils.NotSpecF,
                            help='site-type 2 worker multipliers for w1, w2, w3 (missing args set to 1)')
        parser.add_argument('--site3_mult_workers', type=utils.comma_separated_float_3, required=False,
                            default=utils.NotSpecF,
                            help='site-type 3 worker multipliers for w1, w2, w3 (missing args set to 1)')

        parser.add_argument('--site1_exp_time', type=utils.nnint, required=False, default=utils.NotSpecI,
                            help='time multiplier value for all site-type 1')
        parser.add_argument('--site2_exp_time', type=utils.nnint, required=False, default=utils.NotSpecI,
                            help='time multiplier value for all site-type 2')
        parser.add_argument('--site3_exp_time', type=utils.nnint, required=False, default=utils.NotSpecI,
                            help='time multiplier value for all site-type 3')

        # worker properties
        parser.add_argument('--worker1_cost', type=utils.nnfloat, required=False, default=200,
                            help='cost per unit time of worker-type 1')
        parser.add_argument('--worker2_cost', type=utils.nnfloat, required=False, default=400,
                            help='cost per unit time of worker-type 2')
        parser.add_argument('--worker3_cost', type=utils.nnfloat, required=False, default=500,
                            help='cost per unit time of worker-type 3')

        # configuration parameters
        parser.add_argument('--budget', type=utils.nnfloat, required=False, default=999,
                            help='budget - total costs cannot exceed this')
        parser.add_argument('--duration', type=utils.pint, required=False, default=5,
                            help='duration - number of time steps (eg days) in simulation')

        if args_list is None:
            self.args = parser.parse_args()
        else:
            self.args = parser.parse_args(args_list)

        # set defaults to create graph if not loading graph and args not specified
        if self.args.filename_graph == "":
            if self.args.max_x == utils.NotSpecI:
                self.args.max_x = 10
            if self.args.max_y == utils.NotSpecI:
                self.args.max_y = 10
            if self.args.num_verts == utils.NotSpecI:
                self.args.num_verts = 11
            if self.args.graph_type == "":
                self.args.graph_type = 'random'
            if self.args.num_site1 == utils.NotSpecI:
                self.args.num_site1 = 3
            if self.args.num_site2 == utils.NotSpecI:
                self.args.num_site2 = 2
            if self.args.num_site3 == utils.NotSpecI:
                self.args.num_site3 = 4

        self.worker_cost_rate = {
            Worker_Type.WORKER1: self.args.worker1_cost,
            Worker_Type.WORKER2: self.args.worker2_cost,
            Worker_Type.WORKER3: self.args.worker3_cost
        }

        self.graph = self.load_graph(verbose)

    def load_graph(self, verbose=False) -> Graph:
        '''
        Load graph from filename into SiteStructures

            Arguments:
                verbost: bool, default = False

            Return:
                Graph (class)
        '''

        filename_graph = self.args.filename_graph

        # load graph from file if filename
        if len(filename_graph) > 0:

            # set up structures for parameters - site_structures will have arg defaults here
            # if args are unspecified, a loaded graph can override below
            self.site_structures = Site_Structures(
                self.args, use_arg_defaults=True)

            # set args from graph vars
            #
            # default dict for site access times in class Parameter - graph vars set in Configuration instantiation
            try:
                graph_type = Graph_Type.RANDOM
                graph = Graph(self.args.num_verts, self.args.max_x,
                              self.args.max_y, graph_type)
                graph.load_from_json(filename_graph)
            except Exception as msg_e:
                msg = f'Could not load graph file "{filename_graph}" from current directory {os.getcwd()}: {msg_e}.'
                print(msg, flush=True)
                raise RuntimeError(msg)

            #
            # graph parameters will be overridden below if args are specified
            #
            # noinspection DuplicatedCode
            for vt in self.site_structures.site_acquire_times.keys():
                vt_list = graph.get_vertices_type(vt)
                if len(vt_list) > 0:
                    if verbose:
                        print(
                            f'in graph, vertex type {vt} acq time = {vt_list[0].time_to_acquire}', flush=True)
                        if self.site_structures.site_acquire_times[vt] != utils.NotSpecI:
                            print(
                                f'   ...overriding graph value with command-line arg value of acquire times')
                else:
                    if verbose:
                        print(f'no vertices found in graph of type {vt}')

            for vt in self.site_structures.site_rewards.keys():
                vt_list = graph.get_vertices_type(vt)
                if len(vt_list) > 0:
                    if verbose:
                        print(
                            f'in graph, vertex type {vt} reward = {vt_list[0].reward}', flush=True)
                        if self.site_structures.site_rewards[vt] != utils.NotSpecF:
                            print(
                                f'   ...overriding graph value with command-line arg value of rewards')
                else:
                    if verbose:
                        print(f'no vertices found in graph of type {vt}')

            for vt in self.site_structures.site_mult_time.keys():
                vt_list = graph.get_vertices_type(vt)
                if len(vt_list) > 0:
                    if verbose:
                        print(
                            f'in graph, vertex type {vt} multiplier time = {vt_list[0].mult_time}', flush=True)
                        if self.site_structures.site_mult_time[vt] != utils.NotSpecF:
                            print(
                                f'   ...overriding graph value with command-line arg value of time multiplier')
                else:
                    if verbose:
                        print(f'no vertices found in graph of type {vt}')

            # noinspection DuplicatedCode
            for vt in self.site_structures.site_mult_time_active.keys():
                vt_list = graph.get_vertices_type(vt)
                if len(vt_list) > 0:
                    if verbose:
                        print(
                            f'in graph, vertex type {vt} multiplier time active = {vt_list[0].mult_time}', flush=True)
                        if self.site_structures.site_mult_time_active[vt] != utils.NotSpecI:
                            print(
                                f'   ...overriding graph value with command-line arg value of time multiplier active')
                else:
                    if verbose:
                        print(f'no vertices found in graph of type {vt}')

            for vt in self.site_structures.site_mult_worker.keys():
                vt_list = graph.get_vertices_type(vt)
                if len(vt_list) > 0:
                    if verbose:
                        print(
                            f'in graph, vertex type {vt} multiplier worker = {vt_list[0].mult_worker}', flush=True)
                        if self.site_structures.site_mult_worker[vt] != utils.NotSpecF:
                            print(
                                f'   ...overriding graph value with command-line arg value of worker multiplier')
                else:
                    if verbose:
                        print(f'no vertices found in graph of type {vt}')

            for vt in self.site_structures.site_expiration_times.keys():
                vt_list = graph.get_vertices_type(vt)
                if len(vt_list) > 0:
                    if verbose:
                        print(
                            f'in graph, vertex type {vt} exp time = {vt_list[0].expiration_time}', flush=True)
                        if self.site_structures.site_expiration_times[vt] != utils.NotSpecI:
                            print(
                                f'   ...overriding graph value with command-line arg value of expiration times')
                else:
                    if verbose:
                        print(f'no vertices found in graph of type {vt}')

        else:  # graph filename not specified - create graph

            # set up structures for parameters -
            # if args are unspecified, use Site_Structures defaults
            self.site_structures = Site_Structures(
                self.args, use_arg_defaults=False)

            if self.args.num_verts < self.args.num_site1 + self.args.num_site2 + self.args.num_site3 + 1:
                msg = f'Not enough vertices on graph: ' + \
                      f'require number vertices > num site 1 ({self.args.num_site1}) ' + \
                      f'+ num site 2 ({self.args.num_site1}) ' + \
                      f'+ num site 3 ({self.args.num_site3})+ 1'
                print(msg, flush=True)
                raise Exception(msg)
            else:
                if self.args.graph_type == 'random':
                    graph_type = Graph_Type.RANDOM
                else:
                    graph_type = Graph_Type.GRID

                graph = Graph(self.args.num_verts, self.args.max_x,
                              self.args.max_y, graph_type)

                graph.make_graph_connected()

                graph.set_random_sites_origin(
                    self.args.num_site1, self.args.num_site2, self.args.num_site3)

        # above,
        #   graph loaded -> SiteStructure variables were set to values in graph IF arg was unspecified
        #                       S_S vars are only meaningful in this case if sites of same type have identical values
        #                     - if arg was specified, arg value will be written to graph below
        #
        #   graph not loaded -> nothing done to Site_Structure variables
        #                        - will have Site_Structure defaults if arg was unspecified
        #
        # check for argument overrides in parser.add_argument above
        # if the argument has a default value, it was not specified - so don't override
        for v in graph.vertices:
            vt = v.vertex_type
            if vt in v.accessible_types():
                if self.site_structures.site_acquire_times[vt] != utils.NotSpecI:
                    v.time_to_acquire = self.site_structures.site_acquire_times[vt]
                if self.site_structures.site_rewards[vt] != utils.NotSpecF:
                    v.reward = self.site_structures.site_rewards[vt]
                if self.site_structures.site_mult_time[vt] != utils.NotSpecF:
                    v.mult_time = self.site_structures.site_mult_time[vt]
                if self.site_structures.site_mult_time_active[vt] != utils.NotSpecI:
                    v.mult_time_active = self.site_structures.site_mult_time_active[vt]
                if self.site_structures.site_mult_worker[vt] != utils.NotSpecF:
                    v.mult_worker = self.site_structures.site_mult_worker[vt]
                if self.site_structures.site_expiration_times[vt] != utils.NotSpecI:
                    v.expiration_time = self.site_structures.site_expiration_times[vt]

        # copy possible overridden values to args
        args_temp = deepcopy(self.args)
        self.site_structures.values_to_args(args_temp)
        # remove unspecified args
        d = args_temp.__dict__
        d2 = {}
        for k in d.keys():
            if str(d[k]) not in [str(utils.NotSpecI), str(utils.NotSpecF), ""]:
                d2[k] = d[k]
        args_temp.__dict__ = d2
        self.args_trial = args_temp

        return graph

    def values_to_args(self) -> List:
        """
        Arguments:
                args - from argparse.ArgumentParser copy class variables to these args
        Return:
                arg_list - command-line-style argument list that can be
                        used as input to Process_Args()
        """

        arg_list = []
        d = self.args_trial.__dict__
        for k in d.keys():
            arg_list += [f'--{k}', str(d[k])]
        return arg_list

    def save(self, filename: str = None):
        """
        Method to save arguments dict as json
            Arguments:
                filename: str, path to file to save

            Return:
                None
        """
        if filename is None:
            timestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.txt")
            filename = self.args.trial_name + timestr
        with open(filename, 'w') as f:
            json.dump(self.args_trial.__dict__, f)

    @classmethod
    def load(cls, arguments_file: str = None, graph_file: str = None):
       """
        Method to load arguments
            Arguments:
                filename: str, path to file with arguments

                graph_file: str, path to file with graph

            Return:
                list, list of arguments
        """

        def json_to_list(x):
            lst_tup = [('--' + k, str(v)) for k, v in args_dict.items()]
            return list(itertools.chain(*lst_tup))

        args_dict = {}
        args_list = None

        if arguments_file is not None:
            if os.path.isfile(arguments_file):
                with open(arguments_file) as f:
                    args_dict = json.load(f)
            else:
                msg = f'Arguments file {arguments_file} was not found.'
                raise Exception(msg)

        if graph_file is not None:
            if os.path.isfile(graph_file):
                args_dict['filename_graph'] = graph_file
            else:
                msg = f'Graph file {graph_file} was not found.'
                raise Exception(msg)

        args_list = json_to_list(args_dict)
        return cls(args_list)


# set up structures used for problem classes
class Site_Structures:

    def __init__(self, args, use_arg_defaults=True):
        """
        Set up structures used for problem classes
                Note: properties below can be set/varied at SITEs individually

            Arguments:
                args - from argparse.ArgumentParser from which to get values

                use_arg_defaults -
                    True: use defaults in parser.add_argument() below
                    False: use defaults specified here - these defaults override NotSpecI and NotSpecF values
            Return:
                None
        """

        ad = use_arg_defaults

        # default dict for site access times in class Parameter
        self.site_acquire_times = {
            Vertex_Type.SITE1: utils.checki(args.site1_acquire_time, default=1, use_arg_default=ad),
            Vertex_Type.SITE2: utils.checki(args.site2_acquire_time, default=2, use_arg_default=ad),
            Vertex_Type.SITE3: utils.checki(
                args.site3_acquire_time, default=3, use_arg_default=ad)
        }

        self.site_rewards = {
            Vertex_Type.SITE1: utils.checkf(args.site1_reward, default=1000, use_arg_default=ad),
            Vertex_Type.SITE2: utils.checkf(args.site2_reward, default=2000, use_arg_default=ad),
            Vertex_Type.SITE3: utils.checkf(
                args.site3_reward, default=3000, use_arg_default=ad)
        }

        self.site_mult_time = {
            Vertex_Type.SITE1: utils.checkf(args.site1_mult_time, default=1, use_arg_default=ad),
            Vertex_Type.SITE2: utils.checkf(args.site2_mult_time, default=1, use_arg_default=ad),
            Vertex_Type.SITE3: utils.checkf(
                args.site3_mult_time, default=1, use_arg_default=ad)
        }

        # default is entire duration
        self.site_mult_time_active = {
            Vertex_Type.SITE1: utils.checkp2(args.site1_mult_time_active, default=(0, args.duration),
                                             use_arg_default=ad),
            Vertex_Type.SITE2: utils.checkp2(args.site2_mult_time_active, default=(0, args.duration),
                                             use_arg_default=ad),
            Vertex_Type.SITE3: utils.checkp2(
                args.site3_mult_time_active, default=(0, args.duration), use_arg_default=ad)
        }

        # values are dict with key,val = worker_type for which multiplier applies, multiplier
        wt1, wt2, wt3 = Worker_Type.WORKER1, Worker_Type.WORKER2, Worker_Type.WORKER3
        self.site_mult_worker = {
            Vertex_Type.SITE1: utils.checkd3([wt1, wt2, wt3], args.site1_mult_workers, [1, 1, 1], use_arg_default=ad),
            Vertex_Type.SITE2: utils.checkd3([wt1, wt2, wt3], args.site2_mult_workers, [1, 1, 1], use_arg_default=ad),
            Vertex_Type.SITE3: utils.checkd3([wt1, wt2, wt3], args.site3_mult_workers, [
                                             1, 1, 1], use_arg_default=ad)
        }

        # check for value = -1 below - set to last time in this case
        self.site_expiration_times = {
            Vertex_Type.SITE1: utils.checki(args.site1_exp_time, default=args.duration, use_arg_default=ad),
            Vertex_Type.SITE2: utils.checki(args.site2_exp_time, default=args.duration, use_arg_default=ad),
            Vertex_Type.SITE3: utils.checki(
                args.site3_exp_time, default=args.duration, use_arg_default=ad)
        }

    # noinspection DuplicatedCode
    def values_to_args(self, args) -> List:
        """
        Return argument list for input into Process_Args()

            Arguments:
                args: from argparse.ArgumentParser copy class variables to these args

            Return:
                list, arg_list: command-line-style argument list that can be
                        used as input to Process_Args()
        """

        try:
            # noinspection DuplicatedCode
            args.site1_acquire_time = self.site_acquire_times[Vertex_Type.SITE1]
            args.site2_acquire_time = self.site_acquire_times[Vertex_Type.SITE2]
            args.site3_acquire_time = self.site_acquire_times[Vertex_Type.SITE3]

            args.site1_reward = self.site_rewards[Vertex_Type.SITE1]
            args.site2_reward = self.site_rewards[Vertex_Type.SITE2]
            args.site3_reward = self.site_rewards[Vertex_Type.SITE3]

            args.site1_mult_time = self.site_mult_time[Vertex_Type.SITE1]
            args.site2_mult_time = self.site_mult_time[Vertex_Type.SITE2]
            args.site3_mult_time = self.site_mult_time[Vertex_Type.SITE3]

            args.site1_mult_time_active = self.site_mult_time_active[Vertex_Type.SITE1]
            args.site2_mult_time_active = self.site_mult_time_active[Vertex_Type.SITE2]
            args.site3_mult_time_active = self.site_mult_time_active[Vertex_Type.SITE3]

            args.site1_mult_workers = self.site_mult_worker[Vertex_Type.SITE1]
            args.site2_mult_workers = self.site_mult_worker[Vertex_Type.SITE2]
            args.site3_mult_workers = self.site_mult_worker[Vertex_Type.SITE3]

            args.site1_exp_time = self.site_expiration_times[Vertex_Type.SITE1]
            args.site2_exp_time = self.site_expiration_times[Vertex_Type.SITE2]
            args.site3_exp_time = self.site_expiration_times[Vertex_Type.SITE3]
        except:
            raise RuntimeError(
                'Input args missing fields in Site_Structures.values_to_args() call.')

        keys_site = [  # site economics properties
            'site1_acquire_time', 'site2_acquire_time', 'site3_acquire_time',
            'site1_reward', 'site2_reward', 'site3_reward',
            'site1_mult_time', 'site2_mult_time', 'site3_mult_time',
            'site1_mult_time_active', 'site2_mult_time_active', 'site3_mult_time_active',
            'site1_mult_workers', 'site2_mult_workers', 'site3_mult_workers',
            'site1_exp_time', 'site2_exp_time', 'site3_exp_time'
        ]
        arg_list = []
        for k in keys_site:
            att = getattr(args, k)
            if isinstance(att, dict):
                v = list(att.values())
                sv = str(v)[1:-1]  # get rid of brackets on list
            elif isinstance(att, tuple) or isinstance(att, list):
                v = list(att)
                sv = str(v)[1:-1]  # get rid of brackets on tuple or list
            else:
                sv = str(att)

            setattr(args, k, sv)  # convert site structure form to arg form
            arg_list += ['--' + k, sv]

        return arg_list
