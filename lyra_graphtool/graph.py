import json
import pickle
import warnings
from copy import deepcopy
from enum import IntEnum
from random import choices, sample, shuffle
from typing import Dict, Generic, List, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import math

import lyra_graphtool.utils as utils
from lyra_graphtool.edge import Edge
from lyra_graphtool.utils import vertices_array
from lyra_graphtool.worker import Worker_Type
from lyra_graphtool.vertex import Vertex, Vertex_Type

Graph = TypeVar('Graph')


class Graph_Type(IntEnum):
    RANDOM = 0
    GRID = 1


# graph object
class Graph(Generic[Graph]):

    def __init__(self, num_vertices, max_x, max_y, site_structures,
                 gr_type=Graph_Type.RANDOM, int_pairs=True):
        
        '''
        ## Graph object set up graph and vertices for process

            Arguments:

                num_vertices:

                max_x:

                max_y:

                site_structures:

                gr_type: Graph_Type.RANDOM, loaded from Graph_Type(class)

                int_pairs: bool, default=True

            Return:

                None
        '''

        self.vertices = []
        self.edges = []
        self.graph_type = gr_type
        self.generate_integer_pairs = int_pairs  # locations (x,y) integer pairs?
        self.site_structures = site_structures

        if num_vertices > max_x * max_y:
            raise ValueError(f'Number of vertices ({num_vertices}) must be < max_x * max_y ({max_x}*{max_y})')

        if gr_type == Graph_Type.RANDOM:

            # make sure generated points are unique
            unique_flag = False
            ctr = 0
            while not unique_flag and ctr < 10000:  # if >10000 tries, do below
                ctr += 1
                if self.generate_integer_pairs:
                    x_rand = choices(range(max_x), k=num_vertices)
                    y_rand = choices(range(max_y), k=num_vertices)
                else:  # generate real-valued x,y
                    x_rand = np.random.uniform(0, max_x, num_vertices)
                    y_rand = np.random.uniform(0, max_y, num_vertices)

                t = [(x_rand[i], y_rand[i]) for i in range(num_vertices)]

                unique = {}
                for p in t:
                    unique[p] = 1

                unique_flag = len(unique) == num_vertices

            # if one could not generate integer pairs, use sample approach
            #
            if not unique_flag:
                sample_space = [[i, j] for i in range(max_x) for j in range(max_y)]
                shuffle(sample_space)
                x_rand = [sample_space[i][0] for i in range(num_vertices)]
                y_rand = [sample_space[i][1] for i in range(num_vertices)]

            for i in range(num_vertices):
                v = Vertex(x_rand[i], y_rand[i], Vertex_Type.BASIC)
                self.vertices.append(v)

        else:  # grid graph

            radius = np.floor(np.sqrt(num_vertices))

            radius = int(min(radius, max_x + 1, max_y + 1))

            # fill up a cube first
            for x in range(radius):
                for y in range(radius):
                    v = Vertex(x, y, Vertex_Type.BASIC)
                    self.vertices.append(v)

            # fill up right side, then top with rest
            remainder = num_vertices - radius ** 2

            for y in range(min(remainder, radius)):
                v = Vertex(radius, y, Vertex_Type.BASIC)
                self.vertices.append(v)

            remainder_horiz = remainder - radius

            if remainder_horiz > 0:
                y = radius
                for offset in range(remainder_horiz):
                    x = radius - offset
                    v = Vertex(x, y, Vertex_Type.BASIC)
                    self.vertices.append(v)

        #
        # set up edges for travel: in_graph() method
        # defined in Edge class
        #
        self.set_edges()

    # return numpy array of vertices v: v = [ [x0,y0], [x1,y1], ... ]
    def vertices_array(self) -> np.ndarray:
        '''
        ## Method to get numpy array of vertices v: v = [ [x0,y0], [x1,y1], ... ]

            Arguments:
                None

            Return:
                np.ndarray, vertices
        '''
        return vertices_array(self.vertices)

    # get info about vertices: coords and type
    # if no arg passed, get info about all vertices in graph
    def vertices_info(self, vert_list: List = None) -> Dict:
        '''
        ## Get info about vertices: coordinates and type. If no arg passed, get info about all vertices in graph.

            Arguments:
                vert_list: list, default = None
                    list of vertices to recieve info about

            Return:
                dict, vertice information
        '''
        info = {}
        if vert_list is None:
            vert_list = self.vertices
        none_ct = 0
        for v in vert_list:
            if v is not None:
                ct = 1
                if (v.x, v.y) in info.keys():
                    ct = info[(v.x, v.y)][1] + 1
                info[(v.x, v.y)] = (v.vertex_type, ct)
            else:  # None
                none_ct += 1

            if none_ct > 0:
                info['None'] = none_ct

        return info

    # get vertices of specific type
    def get_vertices_type(self, v_type: Vertex_Type) -> List:
        '''
        ## Get vertices of a specific type

            Arguments:
                v_type: Vertex_Type (Class)

            Return:
                list, vertices of a given type
        '''
        # make sure type is legitimate
        if v_type not in Vertex_Type:
            types = list(Vertex_Type.__members__)
            raise ValueError(f'vertex type {v_type} is not in vertex_type.{types}')

        # get vertices of given type
        verts = []
        for v in self.vertices:
            if v.vertex_type == v_type:
                verts.append(v)

        return verts

    # get vertex with specified coordinates (x,y)
    def get_vertex_xy(self, x: float, y: float) -> Vertex:
        '''
        ## Get vertex with specified coordinates (x,y)

            Arguments:
                x: float, x vertice value

                y: float, y vertice value

            Return:
                Vertex (Class)
        '''
        for v in self.vertices:
            if v.x == x:
                if v.y == y:
                    return v
        return None

    # set the type of vertex at coords (x,y)
    def set_vertex_type(self, v_type: Vertex_Type, v: Vertex = None, x: float = None, y: float = None) -> Vertex:
        '''
        ## Set the type of vertex as given coordinates (x,y)

            Arguments:
                v_type: Vertex_Type (Class)

                v: Vertex (class), default = None

                x: float, x vertice value, default = None

                y: float, y vertice value, default = None

            Return:
                Vertex (class)
        '''

        if v is None:
            if x is None or y is None:
                raise ValueError('must specify either vertex or (x,y)')
            else:
                v = Vertex(x, y)

        for i in range(len(self.vertices)):
            if self.vertices[i] == v:
                self.vertices[i].vertex_type = v_type
                return self.vertices[i]

        return None

    # set the type of vertex at coords (x,y)
    def set_vertex_coords(self, v: Vertex, x: float, y: float) -> Vertex:
        '''
        ## Set the type of vertex as coordinates (x,y)

            Arguments:
                v: Vertex (class)

                x: float, x vertice value

                y: float, y vertice value

            Return:
                list, vertices of a given type
        '''

        for i in range(len(self.vertices)):
            if self.vertices[i] == v:
                self.vertices[i].x = x
                self.vertices[i].y = y
                return self.vertices[i]

        return None
    
    # set vertex on graph
    def add_vertex(self, v: Vertex):
        
        # check if vertex not in graph already
        for graph_v in self.vertices:
            if graph_v.x == v.x and graph_v.y == v.y:
                raise ValueError("Vertex already in the graph")
        
        # set other parameters for vertex
        v.time_to_acquire = self.site_structures.site_acquire_times[v.vertex_type]
        v.reward = self.site_structures.site_rewards[v.vertex_type]
        v.mult_time = self.site_structures.site_mult_time[v.vertex_type]
        v.mult_time_active = self.site_structures.site_mult_time_active[v.vertex_type]
        v.mult_worker = self.site_structures.site_mult_worker[v.vertex_type]
        v.expiration_time = self.site_structures.site_expiration_times[v.vertex_type]
        self.vertices.append(v)
        self.make_graph_connected()
        self.set_edges()
    
    # remove vertex from graph
    def remove_vertex(self, x: float, y: float):
        for i, v in enumerate(self.vertices):
            if v.x == x and v.y == y:
                self.vertices.pop(i)
        self.set_edges()
                
    def _set_site_structures(self):
        for v in self.vertices:
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

    # closest vertices to specified vertex
    def closest_vertices(self, v: Vertex) -> List:
        # get distances to other vertices
        dist = []
        for v2 in self.vertices:
            d = Edge(v, v2).length
            if d > 0:
                dist.append(d)

        dist_min = min(dist)

        verts_closest = []
        for v2 in self.vertices:
            if Edge(v, v2).length == dist_min:
                verts_closest.append(v2)

        return verts_closest

    # get isolated vertices - those with no nearest neighbor
    # Return: list of deep copies of isolated vertices
    def isolated_vertices(self) -> List:
        isol = []
        for v in self.vertices:
            vc = self.closest_vertices(v)
            if len(vc) > 0:
                if Edge(v, vc[0]).nearest_neighbor():
                    continue
                else:
                    isol.append(v)
            else:
                isol.append(v)

        return isol

    # get adjacent vertices to specified vertex via edges
    def adjacent_vertices(self, v: Vertex) -> List:
        edges_v = self.get_edges_at_vertex(v)

        verts_adj = []
        for e in edges_v:
            v1, v2 = e.v1, e.v2
            if v1 != v:
                verts_adj.append(v1)
            else:
                verts_adj.append(v2)

        return verts_adj

    #
    # set up edges for travel: in_graph() method
    # defined in Edge class
    # Note: this method needs to be called if vertices are changed
    #
    def set_edges(self) -> List:

        self.edges = []  # delete old edge set if there was one
        for v in self.vertices:
            for v2 in self.vertices:
                e = Edge(v, v2)
                if e.in_graph():  # does this edge meet the conditions to be in graph?
                    if e not in self.edges:  # may have (v2,v1) in edges
                        self.edges.append(e)
        return self.edges

    # get list of edges in given list
    # if edge list is not given, get info for all edges in graph
    def edges_info(self, edge_list: List = None) -> List:
        info = {}
        if edge_list is None:
            edge_list = self.edges
        for e in edge_list:
            x1, y1 = e.v1.x, e.v1.y
            x2, y2 = e.v2.x, e.v2.y
            coords = ((x1, y1), (x2, y2))
            ct = 1
            if coords in info.keys():
                ct = info[coords] + 1
            info[((x1, y1), (x2, y2))] = ct

        return info

    # return edges containing specified vertex
    def get_edges_at_vertex(self, v: Vertex) -> List:
        edges = []
        for e in self.edges:
            v1, v2 = e.v1, e.v2
            if v == v1 or v == v2:
                edges.append(e)

        return edges

    # randomly set origin and sites
    def set_random_sites_origin(self, n_site1: int, n_site2: int, n_site3: int) -> None:

        num_sites = n_site1 + n_site2 + n_site3
        num_v = len(self.vertices)

        if num_v < num_sites + 1:
            raise ValueError(f'Not enough vertices on graph: require number vertices ({num_v}) ' +
                             f'>= num site 1 ({n_site1}) + num site 2 ({n_site2}) ' +
                             f'+ num site 3 ({n_site3})+ 1'
                             )

        index_verts = sample(range(0, num_v), num_sites + 1)  # sites and origin

        # set sites
        for i in index_verts[:n_site1]:
            self.vertices[i].vertex_type = Vertex_Type.SITE1
        for i in index_verts[n_site1: n_site1 + n_site2]:
            self.vertices[i].vertex_type = Vertex_Type.SITE2
        for i in index_verts[-n_site3 - 1: -1]:
            self.vertices[i].vertex_type = Vertex_Type.SITE3

        # set last to origin
        self.vertices[index_verts[-1]].vertex_type = Vertex_Type.ORIGIN
        self._set_site_structures()

    # return edge of minimum distance between two sets of vertices
    # this is the minimum distance of points (v1,v2) v1 in set1, v2 in set2
    @staticmethod
    def distance(vert_list_1: List, vert_list_2: List) -> Edge:

        edges = []
        for v1 in vert_list_1:
            for v2 in vert_list_2:
                edges.append(Edge(v1, v2))

        edge_lens = [e.length for e in edges]

        min_len = min(edge_lens)

        return edges[edge_lens.index(min_len)]

    # depth first search
    def depth_first_search(self, v: Vertex, visited: List) -> List:

        # mark current vertex as visited
        visited.append(v)

        # repeat for all vertices adjacent to v
        for v2 in self.adjacent_vertices(v):

            if v2 not in visited:
                # update the visited list
                visited = self.depth_first_search(v2, visited)

        return visited

    # connected components
    def connected_components(self) -> List:

        visited = []
        cc = []

        for v in self.vertices:

            if v not in visited:
                comp_v = self.depth_first_search(v, [])
                visited += comp_v
                cc.append(comp_v)

        return cc

    # make graph connected by moving connected components in x then y directions
    # towards the largest component until the number of components is reduced
    def make_graph_connected(self) -> None:

        if len(self.vertices) == 1:
            return

        # get largest component as anchor
        cc = self.connected_components()
        cc_len = [len(c) for c in cc]
        ind_max = cc_len.index(max(cc_len))
        anchor_component = cc[ind_max]
        components = cc[:ind_max] + cc[ind_max + 1:]

        while len(components) > 0:

            c0 = components[0]

            # get vertices of min distance
            e_min = self.distance(anchor_component, c0)

            v1, v2 = e_min.v1, e_min.v2

            if v1 in anchor_component:
                va, v0 = v1, v2
            else:
                va, v0 = v2, v1

            # move c0 towards anchor
            sx = np.sign(va.x - v0.x)
            sy = np.sign(va.y - v0.y)

            cl = len(components)

            c0_translate_x = deepcopy(c0)
            ind_v0 = c0_translate_x.index(v0)
            x_diff = sx
            y_diff = sy

            # keep translating until:
            #   - number of connected components decrease, or
            #   - the set we are translating is directly above or below anchor
            #     ie  sx changes to zero (at this point, we'll translate in y dir)
            while len(components) == cl and x_diff == sx and x_diff != 0:

                # translate x components of c0
                for i in range(len(c0)):
                    c = c0[i]
                    # note  cmin is translated to va + (sx,0)
                    # which is a translation by (vax + sx - v0x)
                    x_new = c.x + sx
                    y_new = c.y
                    self.set_vertex_coords(v=c, x=x_new, y=y_new)  # change graph vert
                    c0_translate_x[i].x, c0_translate_x[i].y = x_new, y_new

                self.set_edges()  # update edges since vertices were changed

                # recompute the number of connected components & anchor
                cc = self.connected_components()
                cc_len = [len(c) for c in cc]
                ind_max = cc_len.index(max(cc_len))
                anchor_component = cc[ind_max]
                components = cc[:ind_max] + cc[ind_max + 1:]

                if len(components) == cl:
                    x0 = c0_translate_x[ind_v0].x
                    x_diff = np.sign(va.x - x0)

            # translate y components of c0_translate_x
            c0_translate_y = deepcopy(c0_translate_x)

            # keep translating until:
            #   - number of connected components decrease, or
            #   - the set we are translating is directly left or right of anchor
            #     ie  sy changes to zero
            while len(components) == cl and y_diff == sy and y_diff != 0:

                for i in range(len(c0_translate_y)):
                    c = c0_translate_y[i]
                    # note  cmin is translated to va + (0,sy)
                    # which is a translation by (vay + sy - v0y)
                    x_new = c.x
                    y_new = c.y + sy
                    self.set_vertex_coords(v=c, x=x_new, y=y_new)
                    c0_translate_y[i].x, c0_translate_y[i].y = x_new, y_new

                self.set_edges()  # update edges since vertices were changed

                # recompute the number of connected components & anchor
                cc = self.connected_components()
                cc_len = [len(c) for c in cc]
                ind_max = cc_len.index(max(cc_len))
                anchor_component = cc[ind_max]
                components = cc[:ind_max] + cc[ind_max + 1:]

                if len(components) == cl:
                    y0 = c0_translate_y[ind_v0].y
                    y_diff = np.sign(va.y - y0)

            if len(components) == cl:
                # neither translation above resulted in the nearest neighboring of sets
                print('after translations in x and y, no reduction in components')
                print('components:')
                for c in components:
                    print(f'{self.vertices_info(c)}')
                return

        #
        # make minimum x and y coords of graph zero
        #
        x_coords = [v.x for v in self.vertices]
        y_coords = [v.y for v in self.vertices]

        x_min = min(x_coords)
        y_min = min(y_coords)

        for v in self.vertices:
            v.x = v.x - x_min
            v.y = v.y - y_min

        self.set_edges()  # update edges since vertices were changed

        return

    # graph vertices
    # color vertices in verts_list OTHER color
    def print_graph(self, vert_list: List = None, filename_out=None) -> None:
        color_dict = {
            'BASIC': 'black',
            'ORIGIN': 'yellow',
            'SITE1': 'red',
            'SITE2': 'blue',
            'SITE3': 'green'
        }

        types = color_dict.keys()

        # plot edges
        for e in self.edges:
            x_values = [e.v1.x, e.v2.x]
            y_values = [e.v1.y, e.v2.y]
            plt.plot(x_values, y_values, linestyle="--", color='black')

        plots = []
        for t in types:
            verts = self.get_vertices_type(Vertex_Type[t])
            x, y = vertices_array(verts).T
            p = plt.scatter(x, y, color=color_dict[t])
            plots.append(p)

        # color vertices in vert_list orange
        if vert_list is not None:
            x, y = vertices_array(vert_list).T
            plt.scatter(x, y, color='orange')

        plt.legend(tuple(plots), tuple(types), scatterpoints=1,
                   bbox_to_anchor=(0.5, -.2),
                   loc='lower center',
                   ncol=3,
                   fontsize=8)

        if filename_out is not None:
            plt.savefig(filename_out)

        plt.show()

    # save graph to file
    def save(self, file_name: str) -> None:
        fn = file_name

        with open(fn, 'wb') as f:
            pickle.dump(self, f)

        # get all paths from vertex v1 to vertex v2
        # return: List of lists, each member list being vertices that represent the path
        # Note:
        #  - adjacent vertices in the list are adjacent in the graph - ie compose an edge
        #  - returned paths do not have cycles - each vertex in the path is visited only once
        #  - max_len argument, if set, will limit paths to this maximum length

    def paths(self, v1: Vertex, v2: Vertex, max_len: float = math.inf, visited: List = None, ) -> List:

        if visited is None:
            visited = []

        visited.append(v1)

        adj_list = self.adjacent_vertices(v1)
        vnext_list = [v for v in adj_list if v not in visited]

        if len(vnext_list) == 0 or len(visited) > max_len:  # nothing left
            return [[]]
        else:
            paths_return = []
            # try next possible paths
            for vnext in vnext_list:  # get adj verts not visited

                visited_next = deepcopy(visited)

                if vnext == v2:
                    paths_next = [[v2]]
                else:
                    paths_next = self.paths(vnext, v2, max_len, visited_next)  # get path vnext to end v2

                for p in paths_next:
                    if len(p) > 0:
                        paths_return.append([v1] + p)  # path from start v1 to end v2

            return paths_return

        # save graph data to json

    def save_to_json(self, file_name: str) -> None:
        graph_json = {}
        graph_json['vertices'] = {}
        for v in self.vertices:  # set up dict
            graph_json['vertices'][str(v.x)] = {}
        for v in self.vertices:  # set up dict
            graph_json['vertices'][str(v.x)][str(v.y)] = {}

        for v in self.vertices:
            for k in vars(v).keys():
                if k != 'x' and k != 'y':
                    val = getattr(v, k)
                    # print(f'v={(v.x,v.y)}  {k}={k}, type k={type(k)}, val={val}, typev={type(val)}')
                    graph_json['vertices'][str(v.x)][str(v.y)][k] = getattr(v, k)

        graph_json['edges'] = []
        for e in self.edges:
            ej = [e.v1.x, e.v1.y, e.v2.x, e.v2.y]
            ejs = [str(val) for val in ej]
            graph_json['edges'].append(ejs)

        # save to json
        with open(file_name, 'w') as f:
            json.dump(graph_json, f)

        # load graph data from json

    def load_from_json(self, file_name: str) -> None:
        # dict to convert integer keys to Worker_Type keys
        wt_dict = {}
        lwt = list(Worker_Type.__members__.values())
        for i in range(len(lwt)):
            wt = lwt[i]
            wt_dict[wt.value] = wt

        # dict to convert integer keys to Vertex_Type keys
        vt_dict = {}
        lvt = list(Vertex_Type.__members__.values())
        for i in range(len(lvt)):
            vt = lvt[i]
            vt_dict[vt.value] = vt

        with open(file_name) as f:
            gj = json.load(f)

        vertsj = gj['vertices']
        edgesj = gj['edges']

        vertices = []
        for xs in vertsj.keys():
            x = float(xs)
            vertsj_x = vertsj[xs]
            for ys in vertsj_x.keys():
                y = float(ys)
                vj = vertsj_x[ys]
                v = Vertex(x, y)

                # set properties
                for k in vj.keys():
                    if k in ('reward', 'mult_time'):
                        setattr(v, k, float(vj[k]))

                    elif k in ('expiration_time', 'time_to_acquire'):
                        if vj[k] is None:
                            val = None
                        else:
                            val = int(vj[k])

                        setattr(v, k, val)

                    elif k == 'vertex_type':
                        setattr(v, k, vt_dict[int(vj[k])])

                    elif k == 'mult_worker':
                        # create Worker_Type keys
                        mult_worker = {}
                        for kwt in vj[k].keys():
                            mult_worker[wt_dict[int(kwt)]] = vj[k][kwt]
                        setattr(v, k, mult_worker)

                    elif k == 'mult_time_active':
                        setattr(v, k, (vj[k][0], vj[k][1]))

                vertices.append(v)

        edges = []
        for ej in edgesj:
            x1, y1 = float(ej[0]), float(ej[1])
            x2, y2 = float(ej[2]), float(ej[3])

            # get v1 and v2
            v1, v2 = None, None
            for v in vertices:
                if v.x == x1 and v.y == y1:
                    v1 = v
                if v.x == x2 and v.y == y2:
                    v2 = v

            if v1 is None:
                raise RuntimeError(f'{(x1, y1)} in json file edges, but not in json file vertices')
            if v2 is None:
                raise RuntimeError(f'{(x2, y2)} in json file edges, but not in json file vertices')

            e = Edge(v1, v2)
            edges.append(e)

        self.vertices = vertices  # no errors, set up verts & edges
        self.edges = edges



