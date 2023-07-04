import warnings
from collections.abc import Sequence

from vitmvt.utils import import_used_modules
from ..builder import PIPELINES, ROOT_PIPELINES


@PIPELINES.register_module()
class Compose(object):
    """Compose a data pipeline with a sequence of transforms.

    Args:
        transforms (list[dict | callable]):
            Either config dicts of transforms or transform objects.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, Sequence)
        # No root registry of PIPELINES in mmcv,
        # need to dfs config to import all used pipelines
        modules = import_used_modules(transforms, 'datasets')
        for module in modules:
            child_registry = module.PIPELINES
            if child_registry.scope not in ROOT_PIPELINES.children:
                child_registry.parent = ROOT_PIPELINES
                ROOT_PIPELINES._add_children(child_registry)
            else:
                warnings.warn(f'{child_registry.scope} is already a child of'
                              f'{ROOT_PIPELINES.name} registry. Please check'
                              f'repeated using of Compose.')

        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = PIPELINES.build(transform)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict, but got'
                                f' {type(transform)}')

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += f'\n    {t}'
        format_string += '\n)'
        return format_string
