from typing import List, Tuple

import numpy as np

BUDGET = 5000
DURATION = 100

# used for defaults to fill in unspecified args
NotSpecI = -99
NotSpecF = -99.


# function to convert list of vertices into numpy array
def vertices_array(vert_list: List) -> np.ndarray:
    x, y = [], []
    for v in vert_list:
        x.append(v.x)
        y.append(v.y)

    return np.array([x, y]).T  # transpose into ordered pairs


# noinspection SpellCheckingInspection
def grstr(arg):
    a_types = ['random', 'grid', '']
    if arg.lower() not in a_types:
        raise ValueError(f"{arg} must be '{a_types[0]}' or '{a_types[1]}'")
    return arg.lower()


# noinspection SpellCheckingInspection
def sstr(arg):
    a_types = ['dfs', 'bestbound', 'hybrid']
    if arg.lower() not in a_types:
        raise ValueError(f"{arg} must be '{a_types[0]}','{a_types[1]}', or {a_types[2]}")
    if arg == 'dfs':
        return 'DFS'
    elif arg == 'bestbound':
        return 'BestBound'
    else:
        return 'hybrid'


# noinspection SpellCheckingInspection
def bostr(arg):
    a_types = ['cost', 'random', 'none']
    if arg.lower() not in a_types:
        raise ValueError(f"{arg} must be '{a_types[0]}','{a_types[1]}', or {a_types[2]}")
    return arg.lower()


# noinspection SpellCheckingInspection
def ostr(arg):
    a_types = ['roi', 'profit', 'revenue']
    if arg.lower() not in a_types:
        raise ValueError(f"{arg} must be '{a_types[0]}','{a_types[1]}', or {a_types[2]}")
    return arg.lower()


def comma_separated_int_2(arg):
    try:
        int_list = list(map(int, arg.split(",")))  # get arguments into list
    except ValueError:
        raise ValueError(f"{arg} not of form nn1 or nn1,nn2 for nn1 <= nn2 non-negative integers")

    if len(int_list) == 1:
        int_list.append(-1)  # if one arg t, apply time mult from t until last time represented by -1

    elif len(int_list) > 2 or int_list[0] < 0 or int_list[0] > int_list[1]:
        raise ValueError(f"{arg} not of form nn1 or nn1,nn2 for nn1 <= nn2 non-negative integers")

    return int_list


def comma_separated_float_3(arg):
    try:
        float_list = list(map(float, arg.split(",")))  # get arguments into list
    except ValueError:
        raise ValueError(f"{arg} not of form nn1 or nn1,nn2 or nn1,nn2,nn3 for nn1,nn2,nn3 non-negative floats")

    if len(float_list) == 1:
        float_list += [1, 1]  # if one arg t, apply time mult from t until last time represented by -1

    elif len(float_list) == 2:
        float_list.append(1)

    return float_list


def pint(arg):
    try:
        i = int(arg)
    except ValueError:
        raise ValueError(f"{arg} not an integer literal")

    if i <= 0:
        raise ValueError(f"{i} must be positive")
    return i


# noinspection SpellCheckingInspection
def nnint(arg):
    try:
        i = int(arg)
    except ValueError:
        raise ValueError(f"{arg} not an integer literal")

    if i < 0:
        raise ValueError(f"{i} must be non-negative")
    return i


# noinspection SpellCheckingInspection
def nnfloat(arg):
    try:
        x = float(arg)
    except ValueError:
        raise ValueError(f"{arg} not a floating-point literal")

    if x < 0.0:
        raise ValueError(f"{x} must be non-negative")
    return x


# noinspection SpellCheckingInspection
def checki(arg: int, default: int = NotSpecI, use_arg_default: bool = False):
    if arg != NotSpecI or use_arg_default:
        return arg
    else:
        if default != NotSpecI:
            return default
        else:
            msg = f'\n\n***** int argument default not specified in class Site_Structure(). *****\n'
            raise ValueError(msg)


# noinspection SpellCheckingInspection
def checkf(arg: float, default: int = NotSpecF, use_arg_default: bool = False):
    if arg != NotSpecF or use_arg_default:
        return arg
    else:
        if default != NotSpecF:
            return default
        else:
            msg = f'\n\n***** float argument default not specified in class Site_Structure(). *****\n'
            raise ValueError(msg)


# noinspection SpellCheckingInspection
def checkp2(arg, default: Tuple, use_arg_default: bool = False):
    if use_arg_default:
        if arg == NotSpecI:
            return arg  # return int NotSpecI to determine if arg specified
        elif isinstance(arg, list):
            return arg[0], arg[1]
        else:
            msg = f'\n\n***** ordered pair argument default not specified in args. *****\n'
            raise ValueError(msg)

    if arg != NotSpecI:
        return arg[0], arg[1]
    else:
        if default != NotSpecI:
            return default
        else:
            msg = f'\n\n***** ordered pair argument default not specified in class Site_Structure(). *****\n'
            raise ValueError(msg)


# return worker_mult dictionary
# noinspection SpellCheckingInspection
def checkd3(type_list, arg_list, default, use_arg_default: bool = False):
    if use_arg_default:
        if arg_list == NotSpecF:
            return arg_list  # return float NotSpecF to determine if arg specified
        elif isinstance(arg_list, list):
            if len(arg_list) == len(type_list):
                dict_ret = {}
                for i in range(len(type_list)):
                    dict_ret[type_list[i]] = arg_list[i]
                return dict_ret
            else:
                msg = f'\n\n***** args list must be length {len(type_list)} *****\n'
                raise ValueError(msg)

        else:
            msg = f'\n\n***** dictionary argument default not specified in args. *****\n'
            raise ValueError(msg)

    if arg_list != NotSpecF:
        ret_list = arg_list
    else:
        ret_list = default

    dict_ret = {}
    for i in range(len(type_list)):
        dict_ret[type_list[i]] = ret_list[i]

    return dict_ret

