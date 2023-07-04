from vitmvt.space import Space, build_search_space
from .helpers import make_divisible


def dmcp_make_divisible(out_channels, pruning_rate):
    """
    Define the search space of the channel according to the pruning
    rate range, where the search space consists of two parts
        1. sampled by pruning rate (that is, maximum, minimum and random
            pruning rate)
        2. sampled by probability
    """
    if issubclass(type(out_channels), Space):
        out_channels = out_channels()

    min_rate = pruning_rate.lower
    max_rate = pruning_rate.upper
    rate_offset = pruning_rate.step

    # sampled by probability
    group_size = int(rate_offset * out_channels / max_rate)
    num_groups = int((max_rate - min_rate) / rate_offset + 1e-4)
    min_ch = out_channels - (group_size * num_groups)
    assert min_ch > 0
    ch_indice = []
    assert group_size * num_groups + min_ch == out_channels
    for i in range(num_groups + 1):
        ch_indice.append(min_ch + i * group_size)

    # sampled by pruning rate
    sample_rate = [
        round(min_rate + i * rate_offset, 3) for i in range(num_groups + 1)
    ]
    sampled_ch_num = []
    for pr in sample_rate:
        sampled_ch_num.append(int(make_divisible(out_channels * pr / 1.0)))

    data = sorted(list(set(ch_indice + sampled_ch_num)))
    space = dict(type='Categorical', data=data, default=out_channels)
    out_channels_space = build_search_space(space)
    return out_channels_space, (min_ch, num_groups, group_size)
