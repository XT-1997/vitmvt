from abc import abstractmethod

import mmcv
import torch.nn as nn
from orderedset import OrderedSet

from vitmvt.space import FreeBindAPI
from ..mutables import BaseMutable, MixedOp, SliceOp
from ..utils import StructuredMutableTreeNode


class BaseMutator(nn.Module):
    """A mutator is responsible for mutating a graph by obtaining the search
    space from the network and implementing callbacks that are called in
    ``forward`` in mutables.

    Args:
        model (nn.Module): PyTorch model to apply mutator on.
    """

    def __init__(self, model):
        super().__init__()
        self.__dict__['model'] = model
        self._structured_mutables = self._parse_mutables(self.model)
        self._cache = dict()

    def _parse_mutables(self, module, root=None, prefix='', memo=None):
        if memo is None:
            memo = set()
        if root is None:
            root = StructuredMutableTreeNode(None)
        if module not in memo:
            memo.add(module)
            if isinstance(module, BaseMutable):
                module.name = prefix
                module.set_mutator(self)

                if isinstance(module, SliceOp):
                    attrs = vars(module)
                    space_memo = []
                    space_attr = dict()
                    for k, v in attrs.items():
                        if isinstance(v, FreeBindAPI):
                            space_attr[k] = v
                            space_memo.extend(v.get_bind_free_obj())
                    module.space = OrderedSet(space_memo)
                    module.space_attr = space_attr
                root = root.add_child(module)
            for name, submodule in module._modules.items():
                if submodule is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                self._parse_mutables(
                    submodule, root, submodule_prefix, memo=memo)
        return root

    @property
    def mutables(self):
        """A generator of all Mutable modules.

        Modules are yielded in the order that they are defined in ``__init__``.
        For mutables with their keys appearing multiple times, only the first
        one will appear.
        """
        return self._structured_mutables

    @property
    def undedup_mutables(self):
        return self._structured_mutables.traverse(deduplicate=False)

    def forward(self, *args, **kwargs):
        raise RuntimeError('Forward is undefined for mutators.')

    def __setattr__(self, name, value):
        if name == 'model':
            raise AttributeError('Attribute `model` can be set at most once,'
                                 "and you shouldn't use `self.model = model` "
                                 'to include you network, as it will include'
                                 'all parameters in model into the mutator.')
        return super().__setattr__(name, value)

    @abstractmethod
    def sample_search(self, *args, **kwargs):
        """Sample a candidate for training.

        Returns:
            result (dict): The sampled candidaes.
        """
        raise NotImplementedError

    def forward_mixed_op(self, *args, **kwargs):
        """Proxy function for MixedOp."""
        raise NotImplementedError

    def forward_slice_op(self, *args, **kwargs):
        """Proxy function for SliceOp."""
        raise NotImplementedError

    def load_subnet(self, subnet_path, load_type=(MixedOp, SliceOp)):
        """Load subnet searched out in search stage.

        Args:
            subnet_path (str | list[str] | tuple(str) ｜dict variable):
            The path of saved subnet file, its suffix should be .yaml or a dict
            variable which saves the subnet.
            There may be several subnet searched out in some algorithms.

        Returns:
            dict | list[dict]: Config(s) for subnet(s) searched out.
        """
        if subnet_path is None:
            return

        if isinstance(subnet_path, str):
            cfg = mmcv.fileio.load(subnet_path)
        elif isinstance(subnet_path, (list, tuple)):
            cfg = list()
            for path in subnet_path:
                cfg.append(mmcv.fileio.load(path))
        elif isinstance(subnet_path, dict):
            cfg = subnet_path
        else:
            raise NotImplementedError

        return self.load_final_cfg(cfg, load_type)

    def load_final_cfg(self, cfg, load_type=(MixedOp, SliceOp)):
        # convert from cfg to final_cfg
        final_cfg = dict()
        for mutable in self.mutables:
            if isinstance(mutable, MixedOp):
                if MixedOp in load_type:
                    final_cfg[mutable.key] = cfg[mutable.key]
            elif isinstance(mutable, SliceOp):
                if SliceOp in load_type:
                    space_cfg = dict()
                    for space in mutable.space:
                        space_cfg[space.key] = cfg[space.key]
                    final_cfg[mutable.key] = space_cfg
            else:
                raise ValueError('Unsupposed Mutable export for '
                                 f'{mutable.__class__.__name__}')
        return final_cfg
