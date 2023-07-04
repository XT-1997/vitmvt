import copy
from abc import abstractmethod
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn
from mmcv.runner import BaseModule
from torch.nn.utils import clip_grad

from ..builder import build_model


class BaseAlgorithm(BaseModule):

    def __init__(self, model, grad_clip=None, **kwargs):

        super(BaseAlgorithm, self).__init__(**kwargs)
        if isinstance(model, nn.Module):
            self.model = model
        else:
            self.model = build_model(model)
        self.grad_clip = grad_clip

    def forward(self, *args, **kwargs):
        """Simply proxy self.model."""
        return self.model(*args, **kwargs)

    def forward_train(self, *args, **kwargs):
        """Simply proxy self.model."""
        return self.model.forward_train(*args, **kwargs)

    def forward_test(self, *args, **kwargs):
        """Simply proxy self.model."""
        return self.model.forward_dummy(*args, **kwargs)

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.
        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            elif isinstance(loss_value, dict):
                for name, value in loss_value.items():
                    log_vars[name] = value
            # mmpose
            elif isinstance(loss_value, float):
                log_vars[loss_name] = loss_value
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if not isinstance(loss_value, float):
                if dist.is_available() and dist.is_initialized():
                    loss_value = loss_value.data.clone()
                    dist.all_reduce(loss_value.div_(dist.get_world_size()))
                log_vars[loss_name] = loss_value.item()
            else:
                # mmpose
                log_vars[loss_name] = loss_value

        return loss, log_vars

    @abstractmethod
    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training.
        TODO(sensecore): Note that each downstream task may also overide
        `train_step`, which may cause confused. We except a architecture rule
        for different downstream task, to regularize behavior they realized.
        Such as forward propagation, back propagation and optimizer updating.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: Dict of outputs. The following fields are contained.
                - loss (torch.Tensor): A tensor for back propagation, which \
                    can be a weighted sum of multiple losses.
                - log_vars (dict): Dict contains all the variables to be sent \
                    to the logger.
                - num_samples (int): Indicates the batch size (when the model \

                    is DDP, it means the batch size on each GPU), which is \
                    used for averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

        return outputs

    @abstractmethod
    def val_step(self, data, optimizer):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs may
        not implemented with this method, but an evaluation hook.
        """
        pass

    def export(self):
        """Export the current model according to stats and cache.

        Returns:
            nn.Module: Exported model.
        """
        pass

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            grad_clip = copy.deepcopy(self.grad_clip)
            clip_type = grad_clip.pop('clip_type', 'by_norm')
            if clip_type == 'by_norm':
                return clip_grad.clip_grad_norm_(params, **grad_clip)
            elif clip_type == 'by_value':
                return clip_grad.clip_grad_value_(params, **grad_clip)
            else:
                raise ValueError(f'Unsupporsed clip_type: {clip_type}')
