import torch

_counter = 0


def get_module_device(module):
    """Get the device of a module.

    Args:
        module (nn.Module): A module contains the parameters.
    Returns:
        torch.device: The device of the module.
    """
    try:
        next(module.parameters())
    except StopIteration as e:
        raise ValueError('The input module should contain parameters.') from e

    if next(module.parameters()).is_cuda:
        return next(module.parameters()).get_device()

    return torch.device('cpu')


def set_requires_grad(nets, requires_grad=False):
    """Set requires_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def global_mutable_counting():
    """A program level counter starting from 1."""
    global _counter
    _counter += 1
    return _counter


class StructuredMutableTreeNode:
    """A structured representation of a search space.
        A search space comes with a root (with `None` stored in its `mutable`),
        and a bunch of children in its `children`. This tree can be seen as
        a "flattened" version of the module tree.
        TODO: support nested search_space.

    Args:
        mutable (Mutable): The mutable that current node is linked with.
    """

    def __init__(self, mutable):

        self.mutable = mutable
        self.children = []

    def add_child(self, mutable):
        """Add a tree node to the children list of current node."""
        self.children.append(StructuredMutableTreeNode(mutable))
        return self.children[-1]

    def type(self):
        """Return the ``type`` of mutable content."""
        return type(self.mutable)

    def __iter__(self):
        return self.traverse()

    def traverse(self, order='pre', deduplicate=True, memo=None):
        """Return a generator that generates a list of mutables in this tree.

        Args:
            order (str, optional): If pre, current mutable is yield
                before children. Otherwise after.. Defaults to "pre".
            deduplicate (bool, optional): If true, mutables with the same key
                will not appear after the first appearance. Defaults to True.
            memo (dict, optional): An auxiliary dict that memorize keys
                seen before. so that deduplication is possible.
                Defaults to None.

        Yields:
            Mutable: generator of Mutable
        """
        if memo is None:
            memo = set()
        assert order in ['pre', 'post']
        if order == 'pre':
            if self.mutable is not None:
                if not deduplicate or self.mutable.key not in memo:
                    memo.add(self.mutable.key)
                    yield self.mutable
        for child in self.children:
            for m in child.traverse(
                    order=order, deduplicate=deduplicate, memo=memo):
                yield m
        if order == 'post':
            if self.mutable is not None:
                if not deduplicate or self.mutable.key not in memo:
                    memo.add(self.mutable.key)
                    yield self.mutable
