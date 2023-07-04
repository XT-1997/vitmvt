import torch
import torch.nn as nn
from mmcv.cnn.utils.weight_init import trunc_normal_

from ..builder import MUTABLES
from .base_mutable import SliceOp


class ParameterSlice_export(nn.Module):

    def __init__(self, dim1, dim2, max_embedding_dim, cls_token):
        super(ParameterSlice_export, self).__init__()
        self.cls_token = nn.Parameter(
            torch.zeros(dim1, dim2, max_embedding_dim))
        self.cls_token.data = cls_token.data

    def forward(self):
        return self.cls_token


@MUTABLES.register_module()
class ParameterSlice(SliceOp):
    """Sliceable Parameter used in cls_token and pose_embed.

    Args:
        dim1 (int): the first dim of the tensor.
        dim2 (int): the second dim of the tensor.
        embedding_dim (int/Int): The same with nn.Parameter.
    """

    def __init__(self, dim1, dim2, embedding_dim, key=None):
        assert isinstance(embedding_dim.curr,
                          int), 'Only support Transformer Example'
        # Must initionalize first for multiple inheritance.
        SliceOp.__init__(self, key=key)
        max_embedding_dim = self.get_value(embedding_dim, kind='max')
        self.embedding_dim = embedding_dim
        self.cls_token = nn.Parameter(
            torch.zeros(dim1, dim2, max_embedding_dim))
        self.dim1 = dim1
        self.dim2 = dim2
        trunc_normal_(self.cls_token, std=.02)

    def forward_inner(self):
        embedding_dim = self.get_value(self.embedding_dim)
        return self.cls_token[..., :embedding_dim]

    def forward(self, *args, **kwargs):
        """For multiple inheritance, we use the SliceOp's ``forward`` method to
        proxy.

        Note that, each mutiple inheritance mutable should realize this.
        """
        return SliceOp.forward(self, *args, **kwargs)

    def export(self, **kwargs):
        """Export LayerNormSlice to nn.LayerNorm."""
        embedding_dim = kwargs.get('embedding_dim',
                                   self.get_value(self.embedding_dim))

        return ParameterSlice_export(self.dim1, self.dim2, embedding_dim,
                                     self.cls_token[..., :embedding_dim])
