import numpy as np
from mmcv.utils import Registry

FORMATTERS = Registry('formatter')


def build_result_formatter(cfg):
    """build result formatter."""
    return FORMATTERS.build(cfg)


@FORMATTERS.register_module()
class DetResultFormatter:
    """Detection result formatter.

    Args:
        extract_list (bool): If true, extract the first member of the result.
            Used in demo to visualize a single image result. Default: False.
        encode_mask (bool): If true, using RLE to encode mask.
            Default: False.
    """

    def __init__(self, extract_list=False, encode_mask=False):
        self.extract_list = extract_list
        self.encode_mask = encode_mask

    def __call__(self, model, result):
        if self.encode_mask:
            if isinstance(result[0], tuple):
                from mmdet.core import encode_mask_results
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
        if self.extract_list:
            result = result[0]
        return result


@FORMATTERS.register_module()
class ClsResultFormatter:
    """Classification result formatter.

    Change the classification output array to a result dict.
    """

    def __call__(self, model, result):
        """
        formatter function
        Args:
            model (nn.Module): A classifier model.
            result (ndarray): Classification output array.

        Returns:
            result (Dict): Classification result dict.
        """
        pred_score = np.max(result, axis=1)[0]
        pred_label = np.argmax(result, axis=1)[0]
        result = dict(
            pred_label=pred_label,
            pred_score=float(pred_score),
            pred_class=model.CLASSES[pred_label])
        return result
