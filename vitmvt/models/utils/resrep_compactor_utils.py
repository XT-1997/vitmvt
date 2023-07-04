import numpy as np
import torch
import torch.nn.functional as F


def _fuse_kernel(kernel, gamma, running_var, eps):
    std = torch.sqrt(running_var + eps)
    t = gamma / std
    assert t.size(0) == kernel.size(0)
    if (t.size() != kernel.size()):
        t = t.view(-1, 1, 1, 1)
        t = t.repeat(1, kernel.size(1), kernel.size(2), kernel.size(3))
    return kernel * t


def _fuse_bias(running_mean, running_var, gamma, beta, eps, bias=None):
    if bias is None:
        return beta - running_mean * gamma / torch.sqrt(running_var + eps)
    else:
        return beta + (bias - running_mean) * gamma / torch.sqrt(running_var +
                                                                 eps)


def fuse_conv_bn(save_dict, pop_name_set, kernel_name):
    mean_name = kernel_name.replace('.conv.weight', '.bn.running_mean')
    var_name = kernel_name.replace('.conv.weight', '.bn.running_var')
    gamma_name = kernel_name.replace('.conv.weight', '.bn.weight')
    beta_name = kernel_name.replace('.conv.weight', '.bn.bias')
    pop_name_set.add(mean_name)
    pop_name_set.add(var_name)
    pop_name_set.add(gamma_name)
    pop_name_set.add(beta_name)
    mean = save_dict[mean_name]
    var = save_dict[var_name]
    gamma = save_dict[gamma_name]
    beta = save_dict[beta_name]
    kernel_value = save_dict[kernel_name]
    return _fuse_kernel(
        kernel_value, gamma, var, eps=1e-5), _fuse_bias(
            mean, var, gamma, beta, eps=1e-5)


def fold_conv(fused_k, fused_b, thresh, compactor_mat):
    metric_vec = torch.sqrt(torch.sum(compactor_mat**2, axis=(1, 2, 3)))
    filter_ids_below_thresh = torch.where(metric_vec < thresh)[0]

    if len(filter_ids_below_thresh) == len(metric_vec):
        sortd_ids = np.argsort(metric_vec)
        filter_ids_below_thresh = sortd_ids[:-1]

    if len(filter_ids_below_thresh) > 0:
        compactor_mat = np.delete(
            compactor_mat, filter_ids_below_thresh, axis=0)

    kernel = F.conv2d(
        fused_k.permute(1, 0, 2, 3), compactor_mat,
        padding=(0, 0)).permute(1, 0, 2, 3)
    Dprime = compactor_mat.shape[0]
    bias = torch.zeros(Dprime)
    for i in range(Dprime):
        bias[i] = fused_b.dot(compactor_mat[i, :, 0, 0])

    return kernel, bias, filter_ids_below_thresh


def kernel_selector(model):
    """select backbone model kernel's name & parameters

    Args:
        model (nn.module)

    Returns:
        tuple: (kernel_name_list, save_dict).
    """
    kernel_name_list = []
    save_dict = {}
    for k, v in model.state_dict().items():
        v = v.detach()
        if (v.ndim in [2, 4]) and ('compactor.pwc' not in k):
            kernel_name_list.append(k)
        save_dict[k] = v
    return kernel_name_list, save_dict


def kernel_pruning(followers, kernel_name_list, kernel_value, pruned_ids,
                   save_dict):
    """delete selected pruned channel in follower's kernel

    Args:
        followers (list): pruned follower layers decided by succ_strategy.
        kernel_name_list (list): list of kernel's name
        kernel_value (Tensor): current selected pruned kernel
        pruned_ids (Tensor): selected pruned channel of current kernel
        save_dict (dict): kernel parameter dict

    Returns:
        save_dict (dict): modified kernel parameter dict
    """
    for fo in followers:
        fo_kernel_name = kernel_name_list[fo]
        fo_value = save_dict[fo_kernel_name]
        if fo_value.ndim == 4:
            fo_value = np.delete(fo_value, pruned_ids, axis=1)
        else:
            fc_idx_to_delete = []
            num_filters = kernel_value.shape[0]
            fc_neurons_per_conv_kernel = \
                fo_value.shape[1] // num_filters
            base = torch.arange(0, fc_neurons_per_conv_kernel * num_filters,
                                num_filters)
            for i in pruned_ids:
                fc_idx_to_delete.append(base + i)
            if len(fc_idx_to_delete) > 0:
                fo_value = np.delete(
                    fo_value, torch.cat(fc_idx_to_delete, dim=0), axis=1)
        save_dict[fo_kernel_name] = fo_value
    return save_dict


def compactor_convert(model, origin_deps, thresh, pacesetter_dict,
                      succ_strategy, save_path):
    compactor_mats = {}
    for submodule in model.modules():
        if hasattr(submodule, 'conv_idx'):
            compactor_mats[submodule.conv_idx] = \
                submodule.pwc.weight.detach()

    pruned_deps = origin_deps
    cur_conv_idx = -1
    pop_name_set = set()

    kernel_name_list, save_dict = kernel_selector(model)
    for conv_id, kernel_name in enumerate(kernel_name_list):
        kernel_value = save_dict[kernel_name]
        if kernel_value.ndim == 2:
            continue
        fused_k, fused_b = fuse_conv_bn(save_dict, pop_name_set, kernel_name)
        cur_conv_idx += 1
        fold_direct = cur_conv_idx in compactor_mats
        fold_follower = (pacesetter_dict is not None) and \
            (cur_conv_idx in pacesetter_dict) and \
            (pacesetter_dict[cur_conv_idx] in compactor_mats)

        if fold_direct:
            fm = compactor_mats[cur_conv_idx]
        elif fold_follower:
            fm = compactor_mats[pacesetter_dict[cur_conv_idx]]
        else:
            continue
        # judge whether conv_idx in compactor

        fused_k, fused_b, pruned_ids = fold_conv(fused_k, fused_b, thresh, fm)
        pruned_deps[cur_conv_idx] -= len(pruned_ids)
        if len(pruned_ids) > 0 and conv_id in succ_strategy:
            followers = succ_strategy[conv_id]
            if type(followers) is not list:
                followers = [followers]
            save_dict = kernel_pruning(followers, kernel_name_list,
                                       kernel_value, pruned_ids, save_dict)

        save_dict[kernel_name] = fused_k
        save_dict[kernel_name.replace('.weight', '.bias')] = fused_b

    save_dict['deps'] = pruned_deps
    for name in pop_name_set:
        save_dict.pop(name)
    # delete pruned kernel

    final_dict = {}
    for k, v in save_dict.items():
        if 'num_batches' not in k and 'compactor' not in k:
            final_dict.update({k.replace('module.', ''): v})
    # abandon compactor

    torch.save(final_dict, save_path)
