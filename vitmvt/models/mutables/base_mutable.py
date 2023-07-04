import warnings
from abc import abstractmethod
from collections import OrderedDict

import torch.nn as nn

from vitmvt.space import FreeBindAPI
from ..utils import global_mutable_counting


class BaseMutable(nn.Module):
    """Mutable is designed to function as a normal layer, with all necessary
    operators' weights. States and weights of architectures should be included
    in mutator, instead of the layer itself. Mutable has a key, which marks
    the identity of the mutable. This key can be used by users to share
    decisions among different mutables. In mutator's implementation, mutators
    should use the key to distinguish different mutables. Mutables that share
    the same key should be "similar" to each other. Currently the default scope
    for keys is global. By default, the keys uses a global counter from 1 to
    produce unique ids.
    TODO: check reserved keys.

    Args:
        key (str, optional): The key of mutable. Defaults to None.
    """

    def __init__(self, key=None):
        super().__init__()
        if key is not None:
            if not isinstance(key, str):
                key = str(key)
                warnings.warn(
                    "Warning: key \"%s\" is not string, converted to"
                    'string.', key)
            self._key = key
        else:
            self._key = self.__class__.__name__
            self._key += str(global_mutable_counting())
        self.init_hook = self.forward_hook = None

    def __call__(self, *args, **kwargs):
        self._check_built()
        return super().__call__(*args, **kwargs)

    def set_mutator(self, mutator):
        if 'mutator' in self.__dict__:
            raise RuntimeError('`set_mutator` is called more than once. Did '
                               'you parse the search space multiple times? Or'
                               'did you apply multiple fixed architectures?')
        self.__dict__['mutator'] = mutator

    @property
    def key(self):
        """Read-only property of key."""
        return self._key

    @property
    def name(self):
        """After the search space is parsed, it will be the module name " of
        the mutable."""
        return self._name if hasattr(self, '_name') else self._key

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def space(self):
        return self._space

    @space.setter
    def space(self, space):
        self._space = space

    def _check_built(self):
        if not hasattr(self, 'mutator'):
            raise ValueError(
                f'Mutator not set for {self}. You might have forgotten to '
                'initialize and apply your mutator. Or did you initialize '
                'a mutable on the fly in forward pass? Move to `__init__` '
                'so that trainer can locate all your mutables.')

    @abstractmethod
    def export(self, **kwargs):
        """export the Mutable op(Module) to a Fixed and clean op(Module)."""
        pass


class MixedOp(BaseMutable):
    """MixedOp is composed of mutiple candidates op. In each forward,
    it selects one of the ``op_candidates``, then apply it on inputs and
    return results.
    Note: Elements in MixedOp can be modified or deleted.
    Adding more choices is not supported yet.

    Args:
        op_candidates ([nn.Module] or OrderedDict): A module list to be
            selected from. If [nn.Module], attribute ``names`` will be setted
            as its str(idx). If OrderedDict, attribute ``names`` will be
            setted as its keys.
        reduction (str, optional): Policy if multiples are selected.
            If ``none``, a list is returned. ``mean`` returns the average.
            ``sum`` returns the sum. ``concat`` concatenate the list at
            dimension 1. Defaults to "sum".
        return_mask (bool, optional): If ``return_mask``, return output tensor
            and a mask. Otherwise return tensor only. Defaults to False.
        key ([type], optional): Key of the input choice. Defaults to None.

    Example:
        # ``op_candidates`` can be [nn.Module] or OrderedDict([nn.Module]).
        >>> self.mixed_op = MixedOp(OrderedDict([
        >>>     ("conv3x3", nn.Conv2d(16, 128, 3)),
        >>>     ("conv5x5", nn.Conv2d(16, 128, 5)),
        >>>     ("conv7x7", nn.Conv2d(16, 128, 7))
        >>> ]))
        >>> del self.mixed_op["conv5x5"]
        >>> self.mixed_op["conv3x3"] = nn.Conv2d(16, 128, 3, stride=2)
    Raises:
        TypeError: Unsupported op_candidates type, Only support
            [nn.Module] or OrderedDict([nn.Module]).
    """

    def __init__(self,
                 op_candidates,
                 reduction='sum',
                 return_mask=False,
                 key=None):
        super().__init__(key=key)
        self.names = []
        if isinstance(op_candidates, OrderedDict):
            for name, module in op_candidates.items():
                assert name not in ['reduction', 'return_mask', '_key',
                                    'key', 'names'], "Please don't use " \
                    f"a reserved name '{name}' for your candidate."
                self.add_module(name, module)
                self.names.append(name)
        elif isinstance(op_candidates, list):
            for i, module in enumerate(op_candidates):
                self.add_module(str(i), module)
                self.names.append(str(i))
        # TODO: support slimmable ops.
        else:
            raise TypeError(
                f'Unsupported op_candidates type: {type(op_candidates)}')
        self.reduction = reduction
        self.return_mask = return_mask

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self._modules[idx]
        return list(self)[idx]

    def __setitem__(self, idx, module):
        key = idx if isinstance(idx, str) else self.names[idx]
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in self.names[idx]:
                delattr(self, key)
        else:
            if isinstance(idx, str):
                key, idx = idx, self.names.index(idx)
            else:
                key = self.names[idx]
            delattr(self, key)
        del self.names[idx]

    def __len__(self):
        return len(self.names)

    def __iter__(self):
        return map(lambda name: self._modules[name], self.names)

    def forward(self, *args, **kwargs):
        out, mask = self.mutator.forward_mixed_op(self, *args, **kwargs)
        if self.return_mask:
            return out, mask
        return out

    def export(self, chosen):
        """Export the Mutable op to a Fixed op.

        Raises:
            RuntimeError: Raise error, if get mismatch choosen.
            ValueError: Raise error, if element in chosen is not 0 or 1.

        Returns:
            nn.Module: Exported module.
        """
        if len(chosen) != len(self):
            raise RuntimeError('Mismatch chosen and MixedOp candidates'
                               f' {len(chosen)} vs {len(self)}.')

        if sum(chosen) == 1 and max(chosen) == 1:
            export_module = self[chosen.index(1)]
        else:
            export_module = nn.ModuleList()
            op_candidates = OrderedDict()
            for idx, name in enumerate(self.names):
                if chosen[idx] not in [0, 1]:
                    raise ValueError(f'Unsupported chosen {chosen[idx]}')
                op_candidates[name] = None if chosen[idx] == 0 else self[name]
            export_module = nn.Sequential(op_candidates)
        return export_module


class SliceOp(BaseMutable):
    """SliceOp is a sliceable Op, containing a super-kernel with which
    weight can be sliced. Different from MixedOp, candidates must have
    the same type, and share with local weights. In each forward,
    it selects one of the candidate, slice the consponding weights and
    (inputs), apply to their common function (such as:
    torch.nn.functional.conv2d).
    Note: The local weights is allocated on the build stage using the `max`
    value in the search_space. Thus, the search_space can only be shrinked.

    Args:
        reduction (str, optional): Policy if multiples are selected.
            If ``none``, a list is returned. ``mean`` returns the average.
            ``sum`` returns the sum. ``concat`` concatenate the list at
            dimension 1. Defaults to "sum".
        return_mask (bool, optional): If ``return_mask``, return output tensor
            and a mask. Otherwise return tensor only. Defaults to False.
        key ([type], optional): Key of the input choice. Defaults to None.
    """

    def __init__(self, reduction='sum', return_mask=False, key=None):
        super().__init__(key=key)
        self.reduction = reduction
        self.return_mask = return_mask

    @property
    def space_attr(self):
        return self._space_attr

    @space_attr.setter
    def space_attr(self, space):
        self._space_attr = space

    @staticmethod
    def get_value(value, kind=None):
        """Get value according to value type and kind.

        Args:
            value (Number / FreeBindAPI): Input value.
            kind (str, optional): Kind to decide the return value.
                Defaults to None.
        """
        # share memory
        if isinstance(value, tuple) and all(value[i] is value[0]
                                            for i in range(1, len(value))):
            data = SliceOp.get_value(value[0], kind)
            return tuple(data for _ in value)
        # the elements in value do not share memory, so wo should use
        # get_value for each element. Used in searching kernel .etc
        # for different dimension
        elif isinstance(value,
                        tuple) and not all(value[i] is value[0]
                                           for i in range(1, len(value))):
            return tuple(SliceOp.get_value(val, kind) for val in value)
        elif isinstance(value, FreeBindAPI):
            return value.data if kind == 'choices' else value(kind=kind)
        else:
            return [value] if kind == 'choices' else value

    def forward(self, *args, **kwargs):
        out, mask = self.mutator.forward_slice_op(self, *args, **kwargs)
        if self.return_mask:
            return out, mask
        return out

    @abstractmethod
    def forward_inner(self, *args, **kwargs):
        """Inner forward of a single path.

        Each slice op should implement this function, and may be called by
        Mutator inner forward_slice_op.
        """
        pass
