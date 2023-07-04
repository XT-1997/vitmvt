import torch
from torch.nn.modules.batchnorm import _BatchNorm

from ...builder import ALGORITHMS
from ..base_mutable_alg import BaseMutableAlgorithm


@ALGORITHMS.register_module()
class ViT_MVT(BaseMutableAlgorithm):

    def __init__(self,
                 model,
                 mutator,
                 bn_training_mode=False,
                 grad_clip=None,
                 **kwargs):
        super(ViT_MVT, self).__init__(
            model, mutator=mutator, grad_clip=grad_clip, **kwargs)
        self.bn_training_mode = bn_training_mode
        self.model.backbone.init_weights()

    def train_step(self, data, optimizer):
        """The iteration step during training.

        First to random sample a subnet from supernet, then to train the
        subnet.
        """
        if self.retraining:
            outputs = super(ViT_MVT, self).train_step(data, optimizer)
            if self.grad_clip is not None:
                self.clip_grads(self.parameters())
        else:
            self.mutator.sample_search(kind='random')
            outputs = super(ViT_MVT, self).train_step(data, optimizer)
            if self.grad_clip is not None:
                self.clip_grads(self.parameters())
        return outputs

    def val_step(self, data, optimizer):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs may
        not implemented with this method, but an evaluation hook.
        """
        pass

    def train(self, mode=True):
        """Overwrite the train method in `nn.Module` to set `nn.BatchNorm` to
        training mode when model is set to eval mode when
        `self.bn_training_mode` is `True`.

        Args:
            mode (bool): whether to set training mode (`True`) or evaluation
                mode (`False`). Default: `True`.
        """
        super(ViT_MVT, self).train(mode)
        if not mode and self.bn_training_mode:
            for module in self.modules():
                if isinstance(module, _BatchNorm):
                    module.training = True
