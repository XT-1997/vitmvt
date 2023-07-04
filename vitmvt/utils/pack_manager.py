from mmcv.utils import import_modules_from_strings

DEFAULT_SCOPE = None


def import_used_modules(cfg, module_name):
    """
    Deep-First-Search the config and import modules from libraries.
    Args:
        cfg (list|dict): Config node.
        module_name (str): Name of the module

    Returns:
        list: A list of imported modules.
    """
    scopes = set()

    def _dfs_cfg(config):
        if isinstance(config, list):
            for child in config:
                _dfs_cfg(child)
            return
        if not isinstance(config, dict):
            return
        type_name = config.get('type', None)
        if type_name and '.' in type_name:
            scope = type_name.split('.')[0]
            # not import mme and mmcv to avoid circular import
            if scope not in ['mme', 'mmcv']:
                scopes.add(scope)
        for _, child in config.items():
            _dfs_cfg(child)

    _dfs_cfg(cfg)
    imported_modules = []
    for scope_name in scopes:
        module = import_modules_from_strings(scope_name + '.' + module_name)
        imported_modules.append(module)
    return imported_modules


def add_children_to_registry(root_registry, registry_name, imported_modules):
    for module in imported_modules:
        child_registry = getattr(module, registry_name)
        if child_registry.scope not in root_registry.children:
            child_registry.parent = root_registry
            root_registry._add_children(child_registry)


def replace_default_registry(default_scope, registry_name, module_name):
    module = import_modules_from_strings(
        default_scope + '.' + module_name, allow_failed_imports=True)
    if module is None:
        return None
    if hasattr(module, registry_name):
        replaced_registry = getattr(module, registry_name)
        return replaced_registry
    else:
        return None
