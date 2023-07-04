from copy import deepcopy

from ...utils import get_root_logger
from ..builder import build_mutator
from ..mutables import BaseMutable, MixedOp, SliceOp
from .base import BaseAlgorithm

logger = get_root_logger()


class BaseMutableAlgorithm(BaseAlgorithm):

    def __init__(self,
                 model,
                 mutator,
                 retraining=False,
                 mutable_cfg=None,
                 **kwargs):
        super(BaseMutableAlgorithm, self).__init__(model, **kwargs)
        mutator['model'] = self.model
        self.mutator = build_mutator(mutator)
        self.retraining = retraining
        if self.retraining:
            self.mutable_cfg = self.load_subnet(mutable_cfg)
            _, self.model = self.export(
                export_cfg=self.mutable_cfg, verbose=False)

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
        return self.mutator.load_subnet(subnet_path, load_type)

    def load_final_cfg(self, cfg, load_type=(MixedOp, SliceOp)):
        return self.mutator.load_final_cfg(cfg, load_type)

    def export(self,
               export_cfg=None,
               verbose=False,
               export_type=(MixedOp, SliceOp),
               use_deepcopy=True,
               fixed_mutable=True):
        """Export the current model according to export_cfg.
        Args:
            export_cfg(dict[dict]): Export cfg indicating the fixed model to be
                exported. The first level key of export_cfg is the `key` of
                Mutable. Defaults to None(use `_cache`).
            verbose(bool): Wether to show log info. Defaults to False.

        Returns:
            dict: Human readable export cfg for export model.
            nn.Module: Exported model.
        """
        if export_cfg is None:
            export_cfg = deepcopy(self.mutator._cache)
        else:
            # If the export function of a mutable return itself,
            # the behaviour of forward will depends on the mutator.
            self.mutator._cache = deepcopy(export_cfg)

        # TODO: check key exactly.
        if use_deepcopy:
            export_model = deepcopy(self.model)
        else:
            export_model = self.model
        human_export_cfg = dict()
        if hasattr(export_model, '_modules'):
            self._export_mutables(
                export_model._modules,
                export_cfg,
                human_export_cfg,
                verbose=verbose,
                export_type=export_type,
                fixed_mutable=fixed_mutable)
        if next(self.parameters()).is_cuda:
            export_model.cuda()
        if not fixed_mutable:
            export_model = self
        return human_export_cfg, export_model

    def _export_mutables(self,
                         module,
                         export_cfg,
                         human_export_cfg,
                         verbose=False,
                         export_type=(MixedOp, SliceOp),
                         fixed_mutable=True):
        for name, mutable in module.items():
            if isinstance(mutable, BaseMutable):
                if isinstance(mutable, SliceOp):
                    if SliceOp in export_type:
                        chosen = export_cfg[mutable.key]
                        if verbose:
                            logger.info('Export mutable name: "{}", '
                                        'key: {} with {}'.format(
                                            mutable.name, mutable.key, chosen))
                        for space in mutable.space:
                            space.set_curr(chosen[space.key])
                        # update space_attr according space.
                        space_attr = dict()
                        for k, v in mutable.space_attr.items():
                            space_attr[k] = v()
                        if fixed_mutable:
                            export_mutable = mutable.export(**space_attr)
                            module[name] = export_mutable
                        human_export_cfg[mutable.name] = chosen
                elif isinstance(mutable, MixedOp):
                    if MixedOp in export_type:
                        chosen = export_cfg[mutable.key]
                        if verbose:
                            logger.info('Export mutable name: "{}", '
                                        'key: {} with {}'.format(
                                            mutable.name, mutable.key, chosen))
                        if fixed_mutable:
                            export_mutable = mutable.export(chosen)
                            module[name] = export_mutable
                        human_export_cfg[mutable.name] = chosen
                else:
                    raise ValueError('Unsupposed Mutable export for '
                                     f'{mutable.__class__.__name__}')
            mutable = module[name]        
            if hasattr(mutable, '_modules'):
                self._export_mutables(
                    mutable._modules,
                    export_cfg,
                    human_export_cfg,
                    verbose=verbose,
                    export_type=export_type,
                    fixed_mutable=fixed_mutable)

    def export_final_cfg(self, subnet_cfg):
        """Export final cfg for save and retraining."""
        final_subnet = dict()
        for mutable in self.mutator.mutables:
            if isinstance(mutable, MixedOp):
                final_subnet[mutable.key] = subnet_cfg[mutable.key]
            elif isinstance(mutable, SliceOp):
                final_subnet.update(subnet_cfg[mutable.key])
            else:
                raise ValueError('Unsupposed Mutable export for '
                                 f'{mutable.__class__.__name__}')
        return final_subnet
