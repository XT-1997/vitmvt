import inspect

from mmcv.fileio import dump, load

from .builder import build_search_space
from .space import (Categorical, Dict, List, SimpleSpace, Space,
                    build_search_space_recur)


def choice_to_space(choice):
    if isinstance(choice, (tuple, list)):
        return tuple(choice_to_space(c) for c in choice)
    elif isinstance(choice, Space):
        return choice
    else:
        return Categorical(data=[choice])


def space_from_file(filename):
    """Build search space recursively from file.

    Args:
        filename (str): the path of file(yaml/json/python) \
                that containing space

    Returns:
        space (Space): Created search Space
    """
    dict_cfg = load(filename)
    return build_search_space_recur(dict_cfg)


def dump_search_space(search_space):
    """Dump search space.

    Args:
        search_space (Space): Search Space

    Raises:
        TypeError: search_space must be a SimpleSpace

    Returns:
        dict_cfg (dict): Created dict cfg containing search Space
    """
    if not isinstance(search_space, SimpleSpace):
        raise TypeError('search_space must be a SimpleSpace')

    keys = inspect.signature(search_space.__class__).parameters.keys()
    dict_cfg = {k: getattr(search_space, k) for k in keys}
    dict_cfg['type'] = search_space.__class__.__name__

    return dict_cfg


def dump_search_space_recur(search_space):
    """Dump search space recursively to dict_cfg.

    Args:
        search_space (Space): Search Space

    Raises:
        TypeError: dict_cfg must be a Dict

    Returns:
        dict_cfg (dict): Created Dict that containing space cfg
    """

    def new_node(v):
        if isinstance(v, Dict):
            return dict()
        elif isinstance(v, List):
            return list()
        else:
            return v

    def parse_space(dict_cfg, space):
        if isinstance(space, Dict):
            for k in space:
                v = space[k]
                # At the end of the space dict, leaf node
                if isinstance(v, SimpleSpace):
                    dict_cfg[k] = dump_search_space(v)
                else:
                    # Add a new node
                    sub_dict = new_node(v)
                    parse_space(sub_dict, v)
                    dict_cfg[k] = sub_dict
        elif isinstance(space, List):
            for idx, v in enumerate(space):
                # At the end of the space dict, leaf node
                if isinstance(v, SimpleSpace):
                    dict_cfg.append(build_search_space(v))
                else:
                    # Add a new node
                    sub_dict = new_node(v)
                    parse_space(sub_dict, v)
                    dict_cfg.append(sub_dict)

    if not isinstance(search_space, Dict):
        raise TypeError('dict_cfg must be a Dict')

    dict_cfg = dict()
    parse_space(dict_cfg, search_space)
    return dict_cfg


def space_to_file(space, filename):
    """Dump search space recursively to file.

    Args:
        space (Space): Search Space
        filename (str): the path of file(yaml/json/python) \
                that containing space
    """
    dict_cfg = dump_search_space_recur(space)
    dump(dict_cfg, filename)
