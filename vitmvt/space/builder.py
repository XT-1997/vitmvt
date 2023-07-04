from mmcv.utils import Registry, build_from_cfg


def build_simple_space_from_cfg(cfg, registry, default_args=None):
    """Build search_space from config dict. Handle simple space and nested
    space differently.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        args (obj): Position arguments for `List` search space.
        kwargs (obj): Key arguments for `Dict` search space.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        object: The constructed object.
    """

    if 'type' in cfg and cfg['type'] in ('List', 'Dict'):
        raise KeyError(f"Unsupported type `{cfg['type']}`,"
                       'please use `build_search_space_recur`')
    return build_from_cfg(cfg, registry, default_args=default_args)


def build_search_space(cfg):
    return SEARCH_SPACE.build(cfg)


SEARCH_SPACE = Registry('search_space', build_func=build_simple_space_from_cfg)
