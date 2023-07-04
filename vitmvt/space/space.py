import itertools
import math
import operator
import random
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from collections.abc import Iterable

import numpy as np
from mmcv.utils import deprecated_api_warning
from orderedset import OrderedSet

from .builder import SEARCH_SPACE, build_search_space

_space_counter = 0


def global_space_counting():
    """A program level counter starting from 1."""
    global _space_counter
    _space_counter += 1
    return _space_counter


class BindCallable(object):
    """Bind Callable bop between lhs and rhs.
    Args:
        fn (function, optional): Function used for identifing
        the search space is binded or not. If None, it will be
        regarde as a free search space (not binded), otherwise,
        binded. Defaults to None.

    Attrs:
        lhs (SimpleSpace): Left hand operation. Must be provided on
            `_new_obj_binary` and `_new_obj_unary`. Defaults to None.
        rhs (SimpleSpace / constant): Rright hand operation. Defaults
            to None.
        op (Function or operators): Oprtators accept lhs and (rhs) as
            input and decides how to hand elements in the lhs and rhs.
        data (Iterable): Search space elements. Defaults to None.
    """

    supported_kinds = [None, 'curr', 'default', 'random', 'min', 'max']

    def __init__(self, fn=None):
        self.fn = fn
        self.lhs = None
        self.rhs = None
        self.op = None
        self.data = None

    @staticmethod
    def _new_obj_binary(lhs, rhs, op, **kwargs):
        """Binary ops make operation on synthesis data (elements)"""
        if isinstance(lhs, SimpleSpace):
            if not isinstance(lhs.data, Iterable):
                raise RuntimeError(
                    'To bind search space should has iterable data attr')
            ldata = lhs.data
        else:
            ldata = [lhs]
        if isinstance(rhs, SimpleSpace):
            if not isinstance(rhs.data, Iterable):
                raise RuntimeError(
                    'To bind search space should has iterable data attr')
            rdata = rhs.data
        else:
            rdata = [rhs]
        bdata = [op(x, y) for x, y in itertools.product(ldata, rdata)]

        def fn(lhs, rhs, op, data, kind=None):
            if kind in [None, 'default', 'curr', 'random']:
                lhs_val = lhs(
                    kind=kind) if isinstance(lhs, SimpleSpace) else lhs
                rhs_val = rhs(
                    kind=kind) if isinstance(rhs, SimpleSpace) else rhs
                out = op(lhs_val, rhs_val)
            elif kind == 'min':
                out = min(data)
            elif kind == 'max':
                out = max(data)
            return out

        if isinstance(lhs, Space) and not isinstance(lhs, SimpleSpace):
            raise RuntimeError(
                'Only SimpleSpace support bind, but get {lhs.__name__}')
        if isinstance(rhs, Space) and not isinstance(rhs, SimpleSpace):
            raise RuntimeError(
                'Only SimpleSpace support bind, but get {rhs.__name__}')
        # Note that the `key is not inheriated
        new_obj = FreeBindAPI(fn=fn)
        new_obj.lhs, new_obj.rhs, new_obj.op = lhs, rhs, op
        new_obj.data = bdata
        return new_obj

    def get_bind_free_obj(self):
        """Return the original free objs of the bind obj."""

        def get_bind_free(obj, free_objs):
            if hasattr(obj, 'fn') and obj.fn is None:
                free_objs.append(obj)
            if isinstance(obj.lhs, FreeBindAPI):
                get_bind_free(obj.lhs, free_objs)
            if isinstance(obj.rhs, FreeBindAPI):
                get_bind_free(obj.rhs, free_objs)

        free_objs = []
        get_bind_free(self, free_objs)
        free_objs = list(OrderedSet(free_objs))
        return free_objs

    @staticmethod
    def _new_obj_unary(lhs, op, *args, **kwargs):
        """Unary ops make operation on lhs's return value directly."""
        if not (isinstance(lhs, SimpleSpace)
                and isinstance(lhs.data, Iterable)):
            raise RuntimeError('Only support binding simple search space'
                               'with iterable data attr')

        for d in lhs.data:
            if not isinstance(d, (int, float, complex)):
                raise ValueError('Only support unary operators for numbers')

        def fn(lhs, op, kind=None):
            lhs_val = lhs(kind=kind)
            out = op(lhs_val, *args, **kwargs)
            return out

        # Note that the `key is not inheriated
        new_obj = FreeBindAPI(fn=fn)
        new_obj.lhs, new_obj.op = lhs, op
        # update data to support complex operation
        new_obj.data = [op(x, *args, **kwargs) for x in lhs.data]
        return new_obj

    def bind_call(self, kind=None):
        if kind not in self.supported_kinds:
            raise TypeError(f'Bind state space only support kind with '
                            f'{str(self.upported_kinds)}')
        if not all((self.fn, self.lhs, self.op)):
            raise RuntimeError('Fn, lhs, op has to be setted before call')
        if self.rhs is not None and self.data is not None:
            return self.fn(self.lhs, self.rhs, self.op, self.data, kind=kind)
        else:
            return self.fn(self.lhs, self.op, kind=kind)

    def __add__(self, x):
        return self._new_obj_binary(self, x, operator.add)

    def __radd__(self, x):
        return self.__add__(x)

    def __sub__(self, x):
        return self._new_obj_binary(self, x, operator.sub)

    def __rsub__(self, x):
        return self._new_obj_binary(x, self, operator.sub)

    def __mul__(self, x):
        return self._new_obj_binary(self, x, operator.mul)

    def __rmul__(self, x):
        return self.__mul__(x)

    def __truediv__(self, x):
        return self._new_obj_binary(self, x, operator.truediv)

    def __rtruediv__(self, x):
        return self._new_obj_binary(x, self, operator.truediv)

    def __floordiv__(self, x):
        return self._new_obj_binary(self, x, operator.floordiv)

    def __rfloordiv__(self, x):
        return self._new_obj_binary(x, self, operator.floordiv)

    def __mod__(self, x):
        return self._new_obj_binary(self, x, operator.mod)

    def __rmod__(self, x):
        return self._new_obj_binary(x, self, operator.mod)

    def __pow__(self, x):
        return self._new_obj_binary(self, x, operator.pow)

    def __rpow__(self, x):
        return self._new_obj_binary(x, self, operator.pow)

    def __round__(self, precision=0):
        return self._new_obj_unary(self, round, precision)

    def __abs__(self):
        return self._new_obj_unary(self, operator.abs)


class Space(object):
    """Basic search space describing set of possible values for hyperparameter.

    Args:
        key (str, optional): The key of Space. Defaults to None.
    """

    def __init__(self, key):
        if key is not None:
            if not isinstance(key, str):
                key = str(key)
                warnings.warn(
                    "Warning: key \"%s\" is not string, converted to"
                    'string.', key)
            self._key = key
        else:
            self._key = self.__class__.__name__
            self._key += str(global_space_counting())
        self._curr = self._default if hasattr(self, '_default') else None
        self._frozen = False

    @property
    def frozen(self):
        return self._frozen

    @frozen.setter
    def frozen(self, status):
        self._frozen = status
        self.other_frozen(status)

    @abstractmethod
    def other_frozen(self, status):
        pass

    @property
    def curr(self):
        return self('curr')

    @property
    def default(self):
        return self('default')

    @property
    def key(self):
        """Read-only property of key."""
        return self._key

    def random(self, randgen=None):
        return self('random', randgen=randgen)

    def __call__(self):
        raise NotImplementedError

    def check_config(self, value):
        """check wether the provided config is valid."""
        return True

    def set_curr(self, val):
        self._curr = val


class SimpleSpace(Space):
    """Simple search space describing set of possible values for
    hyperparameter."""

    def __repr__(self):
        reprstr = '{}({})'.format(self.__class__.__name__, str(self._curr))
        return reprstr

    def other_frozen(self, status):
        pass


class FreeBindAPI(SimpleSpace, BindCallable):
    """API for free state simple space and bind state space. Note that child
    object has to set self._default before super() call, and set self.data
    after.

    Args:
        fn (function or None): Bind state space function.
            Default: None
    """

    def __init__(self, *args, fn=None, key=None):
        super(FreeBindAPI, self).__init__(key=key)
        BindCallable.__init__(self, fn=fn)

    @abstractmethod
    def free_repr(self):
        pass

    def bind_repr(self):
        reprstr = self.__class__.__name__ + '('
        # call curr so that binded catogirical could be set correctly
        self(kind='curr')
        # reprstr += ("data=" + str(self.data) + ", ")
        reprstr += ('curr=' + str(self._curr) + ', bind=True)')
        return reprstr

    def __repr__(self):
        if self.fn is None:
            reprstr = self.free_repr()
        else:
            reprstr = self.bind_repr()
        return reprstr

    @abstractmethod
    def free_call(self, kind=None, **kwargs):
        """Free state space callable function. It will sample a choice from the
        search space if kind is not None or `curr`, or `default`.

        Args:
            kind (str/None): Simple type. Default: None.
        """
        pass

    def __call__(self, kind=None, **kwargs):
        if kind not in self.supported_kinds:
            raise TypeError(f'{self.__class__.__name__} only support kind '
                            f'with {str(self.supported_kinds)}, get {kind}')
        if self.frozen:
            # TODO: output notice or force get kind of elements?
            return self._curr
        if self.fn is None:
            if kind == 'default':
                self._curr = self._default
            elif (kind is not None) and (kind != 'curr'):
                self._curr = self.free_call(kind=kind, **kwargs)
        else:
            self._curr = self.bind_call(kind=kind)
        return self._curr

    @abstractmethod
    def free_len(self):
        pass

    def __len__(self):
        if self.fn is not None:
            return len(self.data)
        else:
            return self.free_len()

    def _data_attr_sample_common(self, kind=None, randgen=None, **kwargs):
        """Common sample function for this has iterable data attr and is not
        None."""
        if not isinstance(self.data, Iterable):
            raise RuntimeError(
                'This function can only be used when self.data is iterable')
        assert randgen is None or isinstance(
            randgen, random.Random), 'Invalid randgen as input'
        rand = randgen if randgen else random
        _curr = self._curr
        if kind == 'random':
            prob = kwargs.get('prob', None)
            _curr = rand.choices(self.data, weights=prob)[0]
        elif kind == 'max':
            _curr = max(self.data)
        elif kind == 'min':
            _curr = min(self.data)
        elif kind == 'crossover':
            prob = kwargs.get('prob', 0.5)
            assert 'target' in kwargs, "attribute 'target' must be provided"
            target = kwargs.get('target', None)
            assert isinstance(
                target, self.__class__), "'target' must have the same type"
            target_curr = target()
            if target_curr not in self.data:
                raise ValueError('Target curr not in self.data')
            if rand.random() < prob:
                _curr = target_curr
        elif kind == 'mutate':
            prob = kwargs.get('prob', 0.5)
            if rand.random() < prob:
                candidates = [d for d in self.data if d != self._curr]
                if len(candidates) > 0:
                    _curr = rand.choice(candidates)
        return _curr


@SEARCH_SPACE.register_module()
class Bits(FreeBindAPI):
    """API for gating operations in pruning. Note that this example only
    performs the selection and pass operation based on the index, and does not
    perform the calculation of the channel selection, and the corresponding
    index parameters need to be input.
    TODO: check its realize.

    Args:
        data (built-in objects): the maxnum of channel.
        fn (function or None): Bind state space function.
            Default: None
        frozen (optional): Set as not searchable
            Default: False
    """
    supported_kinds = [None, 'keep', 'remove', 'iter']

    def __init__(self, data, fn=None, frozen=False, key=None):
        super(Bits, self).__init__(key=key)
        self._curr = data
        self.maxnum = data
        assert fn is None, 'Only support fn is None.'
        self.fn = fn
        self.frozen = frozen
        self.to_bits(data)

    def to_bits(self, data):
        """Convert the input data into a list of gate values."""
        self._index = [1 for i in range(data)]

    def __repr__(self):
        if self.fn is None:
            reprstr = self.__class__.__name__ + '('
            reprstr += ('curr=' + str(self._curr))
            return reprstr
        else:
            raise NotImplementedError

    def __call__(self,
                 kind=None,
                 remove_index=None,
                 return_index=False,
                 **kwargs):
        """Call and update Bits. If return_index is True, then it will return a
        list of existing indexes.

        Args:
            remove_index (dict): List of indexes to be removed.
            If not None, the the index selection will be performed
            according to the selected value of the kind
                Default: None
            return_index (bool): Returns a set of integer indexes related
            to the data
                Default: False
        """
        if kind not in self.supported_kinds:
            raise TypeError(f'{self.__class__.__name__} only support kind '
                            f'with {str(self.supported_kinds)}, get {kind}')
        if self.frozen:
            return self._curr
        if kind == 'remove':
            _index = [1 for i in range(self._curr)]
        elif kind == 'keep':
            _index = [1 for i in range(self.maxnum)]
        elif kind == 'iter':
            _index = self._index
        # uddate self._index
        if remove_index is not None:
            for ind in remove_index:
                _index[ind] = 0
            self._index = _index
            self._curr = sum(self._index)
        if return_index:
            out_dices = []
            for i, data in enumerate(self._index):
                if data == 1:
                    out_dices.append(i)
            return out_dices
        else:
            return self._curr

    def __len__(self):
        if self.fn is None:
            return self._curr
        else:
            raise NotImplementedError


@SEARCH_SPACE.register_module()
class Int(FreeBindAPI):
    supported_kinds = [
        None, 'curr', 'default', 'random', 'min', 'max', 'crossover', 'mutate'
    ]

    def __init__(self, lower, upper, default=None, step=1, fn=None, key=None):
        if lower > upper:
            raise ValueError('Invalid space specified by {} and {}'.format(
                lower, upper))
        self.lower = lower
        self.upper = upper
        self._default = default if default else self.lower
        self.step = step
        super(Int, self).__init__(fn=fn, key=key)
        self.data = range(self.lower, self.upper, self.step)

    def free_len(self):
        return len(self.data)

    def free_call(self, kind=None, randgen=None, **kwargs):
        return self._data_attr_sample_common(
            kind=kind, randgen=randgen, **kwargs)

    def free_repr(self):
        reprstr = self.__class__.__name__ + '('
        reprstr += ('lower=' + str(self.lower) + ', ')
        reprstr += ('upper=' + str(self.upper) + ', ')
        reprstr += ('curr=' + str(self._curr) + ', ')
        reprstr += ('step=' + str(self.step)) + ')'
        return reprstr


@SEARCH_SPACE.register_module()
class Real(FreeBindAPI):
    """Real Space.

    Note that there are some difference between kind `max`. In uncountable
    mode, specifically, `no log` and `step is not None` it will returns
    self.upper directly. Otherwise, upper can not be reached.
    """
    supported_kinds = [None, 'curr', 'default', 'random', 'min', 'max']

    def __init__(self,
                 lower,
                 upper,
                 log=False,
                 default=None,
                 step=None,
                 fn=None,
                 key=None):
        if lower > upper:
            raise ValueError('Invalid space specified by {} and {}'.format(
                lower, upper))
        self.lower = lower
        self.upper = upper
        self.log = log
        self.step = step
        self._default = default if default else self.lower
        super(Real, self).__init__(fn=fn, key=key)
        if (not self.log) and (self.step is not None):
            self.data = np.arange(self.lower, self.upper, self.step).tolist()
            self.supported_kinds += ['crossover', 'mutate']

    def free_len(self):
        if (not self.log) and (self.step is not None):
            return len(self.data)
        else:
            return float('inf')

    def free_call(self, kind=None, randgen=None, **kwargs):
        if isinstance(self.data, Iterable):
            return self._data_attr_sample_common(
                kind=kind, randgen=randgen, **kwargs)

        assert randgen is None or isinstance(
            randgen, random.Random), 'Invalid randgen as input'
        rand = randgen if randgen else random
        if kind == 'random':
            if self.log:
                _curr = math.exp(
                    rand.uniform(math.log(self.lower), math.log(self.upper)))
            else:
                _curr = rand.uniform(self.lower, self.upper)
        elif kind == 'max':
            _curr = self.upper
        elif kind == 'min':
            _curr = self.lower
        return _curr

    def free_repr(self):
        reprstr = self.__class__.__name__ + '('
        reprstr += ('lower=' + str(self.lower) + ', ')
        reprstr += ('upper=' + str(self.upper) + ', ')
        reprstr += ('log=' + str(self.log) + ', ')
        reprstr += ('curr=' + str(self._curr) + ', ')
        reprstr += ('step=' + str(self.step)) + ')'
        return reprstr


@SEARCH_SPACE.register_module()
class Categorical(FreeBindAPI):
    """Search space for hyperparameters which are categorical. Such a
    hyperparameter takes one value out of the discrete set of provided options.

    Args:
        data (list[built-in objects]): The choice candidates
        default (optional): Defaults to None.
        fn (optional): Defaults to None.
    """
    supported_kinds = [
        None, 'curr', 'default', 'random', 'min', 'max', 'crossover', 'mutate'
    ]

    @deprecated_api_warning({
        'value': 'data',
    }, cls_name='Categorical')
    def __init__(self, data, default=None, fn=None, key=None):

        self._default = default if default else data[0]
        assert self._default in data
        super(Categorical, self).__init__(fn=fn, key=key)
        self.data = list(data)

    def __iter__(self):
        if self.fn is not None:
            raise RuntimeError('Bind state Categorical not support __iter__')
        for value in self.data:
            yield value

    def free_len(self):
        return len(self.data)

    def free_call(self, kind=None, randgen=None, **kwargs):
        return self._data_attr_sample_common(
            kind=kind, randgen=randgen, **kwargs)

    def free_repr(self):
        reprstr = self.__class__.__name__ + '('
        for d in self.data:
            reprstr += (str(d) + ', ')
        reprstr += ('curr=' + str(self._curr) + ')')
        return reprstr


@SEARCH_SPACE.register_module()
class Bool(Categorical):

    def __init__(self, default=False, fn=None, key=None):
        super(Bool, self).__init__(
            data=[True, False], default=default, fn=fn, key=key)


class NestedSpace(Space, metaclass=ABCMeta):
    """Nested search space describing set of possible values for
    hyperparameter.

    It could contains SimpleSpace or NestedSpace or constant.
    """

    def __repr__(self):
        reprstr = self.__class__.__name__ + '(' + repr(self.data) + ')'
        return reprstr

    @property
    def kwspaces(self):
        """Returns a ordered flattened key-value search space."""

        raise NotImplementedError

    # TODO: optimize NestedSpace frozen alg


@SEARCH_SPACE.register_module()
class List(NestedSpace):

    def __init__(self, *data, key=None):
        super(List, self).__init__(key=key)
        if data:
            data = build_search_space_recur(list(data), auto_build=False)
        self.data = [*data]

    def __iter__(self):
        for value in self.data:
            yield value

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, key, index):
        self.data[key] = index

    def append(self, item):
        self.data.append(item)

    def pop(self, index=-1):
        return self.data.pop(index)

    def __len__(self):
        return len(self.data)

    def __call__(self, kind=None, randgen=None, **args):
        ret = []
        target = args.get('target', None)
        if kind == 'crossover':
            assert target is not None, 'target not provided'
            assert isinstance(target, List), 'target is not instance of List'
            assert len(target.data) == len(
                self.data), 'target has mismatch length'
        prob = args.get('prob', None)
        for idx, value in enumerate(self.data):
            if isinstance(value, Space):
                if target:
                    args['target'] = target[idx]
                if prob and not isinstance(prob, float):
                    args['prob'] = prob[idx]
                value = value(kind=kind, randgen=randgen, **args)
            ret.append(value)
        return ret

    @property
    def kwspaces(self):
        kw_spaces = OrderedDict()
        for idx, value in enumerate(self.data):
            k = str(idx)
            if isinstance(value, NestedSpace):
                for sub_k, sub_v in value.kwspaces.items():
                    sub_k = k + '.' + sub_k
                    kw_spaces[sub_k] = sub_v
            elif isinstance(value, Space):
                kw_spaces[k] = value
        return kw_spaces

    def set_curr(self, config):
        if not isinstance(config, list) or len(config) != len(self):
            raise TypeError('Expect list type with the same len with taget')
        for idx in range(len(self)):
            # TODO(shiguang): check config value
            if isinstance(self.data[idx], NestedSpace):
                self.data[idx].set_curr(config[idx])
            elif isinstance(self.data[idx], Space):
                self.data[idx].check_config(config[idx])
                self.data[idx].set_curr(config[idx])
            else:
                self.data[idx] = config[idx]

    def other_frozen(self, status):
        for value in self.data:
            if isinstance(value, Space):
                value.frozen(status)

    def get_bind_free_obj(self):
        """Return the original free objs of the bind obj."""

        free_objs = []
        for value in self.data:
            if isinstance(value, (NestedSpace, FreeBindAPI)):
                fobjs = value.get_bind_free_obj()
                free_objs.extend(fobjs)

        free_objs = list(OrderedSet(free_objs))
        return free_objs


@SEARCH_SPACE.register_module()
class Dict(NestedSpace):

    def __init__(self, key=None, **data):
        super(Dict, self).__init__(key=key)
        if data:
            data = build_search_space_recur(data, auto_build=False)
        self.data = dict(**data)

    def __iter__(self):
        for key in self.data:
            yield key

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, data):
        self.data[key] = data

    def __len__(self):
        return len(self.data)

    def __call__(self, kind=None, randgen=None, **args):
        ret = {}
        target = args.get('target', None)
        if kind == 'crossover':
            assert target is not None, 'target not provided'
            assert isinstance(target, Dict), 'target is not instance of Dict'
            assert set(target.data.keys()) == set(
                self.data.keys()), 'target has mismatch keys'
        prob = args.get('prob', None)
        for k, v in self.data.items():
            if isinstance(v, Space):
                if target:
                    args['target'] = target[k]
                if prob and not isinstance(prob, float):
                    args['prob'] = prob[k]
                v = v(kind=kind, randgen=randgen, **args)
            ret[k] = v
        return ret

    @property
    def kwspaces(self):
        kw_spaces = OrderedDict()
        for key, value in self.data.items():
            k = str(key)
            if isinstance(value, NestedSpace):
                for sub_k, sub_v in value.kwspaces.items():
                    sub_k = k + '.' + sub_k
                    kw_spaces[sub_k] = sub_v
            elif isinstance(value, Space):
                kw_spaces[k] = value

        return kw_spaces

    def set_curr(self, config):
        if not isinstance(config, dict):
            raise TypeError(f'Expect dict type, get {type(config)}')
        ck = list(config.keys())
        dk = list(self.data.keys())
        if ck != dk:
            raise KeyError(
                f'Unexcept config key with search_space. {ck} vs {dk}')

        for key, value in self.data.items():
            # TODO(shiguang): check config value
            if isinstance(value, NestedSpace):
                value.set_curr(config[key])
            elif isinstance(value, Space):
                self.data[key].check_config(config[key])
                self.data[key].set_curr(config[key])
            else:
                self.data[key] = config[key]

    def get(self, key, default):
        return self.data.get(key, default)

    def items(self):
        for key in self.data:
            yield key, self.data[key]

    def values(self):
        for key in self.data:
            yield self.data[key]

    def keys(self):
        for key in self.data:
            yield key

    def pop(self, key, default):
        return self.data.pop(key, default)

    def other_frozen(self, status):
        for key in self.data:
            data = self.data[key]
            if isinstance(data, Space):
                data.frozen(status)

    def get_bind_free_obj(self):
        """Return the original free objs of the bind obj."""

        free_objs = []
        for key, value in self.data.items():
            if isinstance(value, (NestedSpace, FreeBindAPI)):
                fobjs = value.get_bind_free_obj()
                free_objs.extend(fobjs)

        free_objs = list(OrderedSet(free_objs))
        return free_objs


def build_search_space_recur(dict_cfg, auto_build=True):
    """Build search space recursively from dict_cfg.

    Args:
        dict_cfg (dict / list): dict or list that containing space cfg

    Raises:
        TypeError: dict_cfg must be a dict

    Returns:
        space (Space): search Space
    """

    def new_node(v):
        if isinstance(v, dict):
            return Dict()
        elif isinstance(v, list):
            return List()
        else:
            return v

    def parse_dict(dict_cfg, space):
        if isinstance(dict_cfg, dict):
            for k, v in dict_cfg.items():
                # At the end of the space dict, leaf node
                if isinstance(v, dict) and 'type' in v.keys() and auto_build:
                    space[k] = build_search_space(v)
                else:
                    # Add a new node
                    sub_space = new_node(v)
                    parse_dict(v, sub_space)
                    space[k] = sub_space
        elif isinstance(dict_cfg, list):
            for idx, v in enumerate(dict_cfg):
                # At the end of the space dict, leaf node
                if isinstance(v, dict) and 'type' in v.keys() and auto_build:
                    space.append(build_search_space(v))
                else:
                    # Add a new node
                    sub_space = new_node(v)
                    parse_dict(v, sub_space)
                    space.append(sub_space)

    if isinstance(dict_cfg, dict):
        search_space = Dict()
    elif isinstance(dict_cfg, list):
        search_space = List()
    else:
        raise TypeError('Can only convert dict to Dict, or list ' +
                        f'to List, get {type(dict_cfg)}')

    parse_dict(dict_cfg, search_space)
    return search_space


@SEARCH_SPACE.register_module()
class Nested(Categorical):
    """Search space for hyperparameters which are categorical. Such a
    hyperparameter takes one value out of the discrete set of provided options.

    Args:
        data (list[built-in objects]): The choice candidates
        default (optional): Defaults to None.
        fn (optional): Defaults to None.
    """

    @deprecated_api_warning({
        'value': 'data',
    }, cls_name='Nested')
    def __init__(self, data, default=None, fn=None, key=None):
        ori_keys = None
        for d in data:
            assert isinstance(d, dict)
            if ori_keys:
                cur_keys = set(d.keys())
                statement = f'Keys must stand same in `Nested` search space! Pair: {cur_keys} {ori_keys}'  # noqa: E501
                assert sorted(cur_keys) == sorted(ori_keys), statement
            else:
                ori_keys = set(d.keys())
        super().__init__(data, default=default, fn=fn, key=key)
