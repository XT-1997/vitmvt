import torch
import torch.nn as nn
from mmcv.cnn.utils.weight_init import trunc_normal_

from ..builder import MUTABLES
from .base_mutable import SliceOp


class RelativePosition2D(nn.Module):
    """Rethinking and Improving Relative Position Encoding for Vision
    Transformer.

    ICCV 2021. https://arxiv.org/pdf/2107.14222.pdf

    Image RPE (iRPE for short) methods are new relative position encoding
    methods dedicated to 2D images.

    Args:
        num_units ([int]): embedding dims of relative position.
        max_relative_position ([int]): The max relative position distance.
    """

    def __init__(self, num_units, max_relative_position=14):
        super().__init__()

        self.num_units = num_units
        self.max_relative_position = max_relative_position
        # The first element in embeddings_table_v is the vertical embedding
        # for the class
        self.embeddings_table_v = nn.Parameter(
            torch.randn(max_relative_position * 2 + 2, num_units))
        self.embeddings_table_h = nn.Parameter(
            torch.randn(max_relative_position * 2 + 2, num_units))

        trunc_normal_(self.embeddings_table_v, std=.02)
        trunc_normal_(self.embeddings_table_h, std=.02)

    def forward(self, length_q, length_k):
        # remove the first cls token distance computation
        length_q = length_q - 1
        length_k = length_k - 1
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        # compute the row and column distance
        distance_mat_v = (
            range_vec_k[None, :] // int(length_q**0.5) -
            range_vec_q[:, None] // int(length_q**0.5))
        distance_mat_h = (
            range_vec_k[None, :] % int(length_q**0.5) -
            range_vec_q[:, None] % int(length_q**0.5))
        # clip the distance to the range of
        # [-max_relative_position, max_relative_position]
        distance_mat_clipped_v = torch.clamp(distance_mat_v,
                                             -self.max_relative_position,
                                             self.max_relative_position)
        distance_mat_clipped_h = torch.clamp(distance_mat_h,
                                             -self.max_relative_position,
                                             self.max_relative_position)

        # translate the distance from [1, 2 * max_relative_position + 1],
        # 0 is for the cls token
        final_mat_v = distance_mat_clipped_v + self.max_relative_position + 1
        final_mat_h = distance_mat_clipped_h + self.max_relative_position + 1
        # pad the 0 which represent the cls token
        final_mat_v = torch.nn.functional.pad(final_mat_v, (1, 0, 1, 0),
                                              'constant', 0)
        final_mat_h = torch.nn.functional.pad(final_mat_h, (1, 0, 1, 0),
                                              'constant', 0)

        final_mat_v = torch.LongTensor(final_mat_v)
        final_mat_h = torch.LongTensor(final_mat_h)
        # get the embeddings with the corresponding distance
        embeddings = self.embeddings_table_v[
            final_mat_v] + self.embeddings_table_h[final_mat_h]

        return embeddings


@MUTABLES.register_module()
class RelativePositionSlice2D(SliceOp):
    """Searchable RelativePosition module.

    Args:
        num_heads (int/Int): Parallel attention heads.
        units ([int/Int]): embedding dims of relative position.
        max_relative_position ([int]): The max relative position distance.
    """

    def __init__(self, num_heads, unit=14, max_relative_position=14, key=None):

        SliceOp.__init__(self, key=key)
        # max_num_heads = self.get_value(num_heads, kind='max')

        max_num_units = unit
        super().__init__()
        self.max_relative_position = max_relative_position

        self.embeddings_table_v = nn.Parameter(
            torch.randn(max_relative_position * 2 + 2, max_num_units))
        self.embeddings_table_h = nn.Parameter(
            torch.randn(max_relative_position * 2 + 2, max_num_units))

        trunc_normal_(self.embeddings_table_v, std=.02)
        trunc_normal_(self.embeddings_table_h, std=.02)
        self.num_heads = num_heads
        self.unit = unit

    def forward_inner(self, length_q, length_k):
        self.sample_head_dim = self.unit
        self.sample_eb_table_h = self.embeddings_table_h[:, :self.
                                                         sample_head_dim]
        self.sample_eb_table_v = self.embeddings_table_v[:, :self.
                                                         sample_head_dim]

        # remove the first cls token distance computation
        length_q = length_q - 1
        length_k = length_k - 1
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        # compute the row and column distance
        distance_mat_v = (
            range_vec_k[None, :] // int(length_q**0.5) -
            range_vec_q[:, None] // int(length_q**0.5))
        distance_mat_h = (
            range_vec_k[None, :] % int(length_q**0.5) -
            range_vec_q[:, None] % int(length_q**0.5))
        distance_mat_clipped_v = torch.clamp(distance_mat_v,
                                             -self.max_relative_position,
                                             self.max_relative_position)
        distance_mat_clipped_h = torch.clamp(distance_mat_h,
                                             -self.max_relative_position,
                                             self.max_relative_position)

        final_mat_v = distance_mat_clipped_v + self.max_relative_position + 1
        final_mat_h = distance_mat_clipped_h + self.max_relative_position + 1
        # pad the 0 which represent the cls token
        final_mat_v = torch.nn.functional.pad(final_mat_v, (1, 0, 1, 0),
                                              'constant', 0)
        final_mat_h = torch.nn.functional.pad(final_mat_h, (1, 0, 1, 0),
                                              'constant', 0)

        final_mat_v = torch.LongTensor(final_mat_v)
        final_mat_h = torch.LongTensor(final_mat_h)
        # get the embeddings with the corresponding distance

        embeddings = self.sample_eb_table_v[
            final_mat_v] + self.sample_eb_table_h[final_mat_h]

        return embeddings

    def forward(self, *args, **kwargs):
        """For multiple inheritance, we use the SliceOp's ``forward`` method to
        proxy.

        Note that, each mutiple inheritance mutable should realize this.
        """
        return SliceOp.forward(self, *args, **kwargs)

    def export(self, **kwargs):
        """Export RelativePositionSlice2D to RelativePosition2D."""
        num_units = self.unit

        export_module = RelativePosition2D(num_units)

        export_module.embeddings_table_v.data.copy_(
            self.embeddings_table_v.data[:, :num_units])
        export_module.embeddings_table_h.data.copy_(
            self.embeddings_table_h.data[:, :num_units])

        return export_module
