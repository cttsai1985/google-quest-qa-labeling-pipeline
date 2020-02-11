from typing import List, Optional, Dict

from .SequenceAugmentation import RandomTruncateHead
from .SequenceAugmentation import RandomTruncateTail
from .SequenceAugmentation import RandomDropWords

from .LabelAugmentation import ClipLabel
from .LabelAugmentation import LabelSoften


def _augmentation_factory(augmentation_gen: str, params: Dict):
    augmentations = globals()
    if augmentation_gen not in augmentations.keys():
        raise NotImplementedError()

    return augmentations[augmentation_gen](**params)


class AugmentationMaster:
    def __init__(self, func_x_list: List, func_y_list: List):
        self.transform_x = self._initialize_func_list(func_x_list)
        self.transform_y = self._initialize_func_list(func_y_list)

    @staticmethod
    def _initialize_func_list(func_list) -> List:
        return [_augmentation_factory(augmentation_gen=func_gen, params=params) for func_gen, params in func_list]

    def transform(self, x=None, y=None):
        if x is not None:
            for tf in self.transform_x:
                x = tf.transform(x)

        if y is None:
            return x

        for tf in self.transform_y:
            y = tf.transform(y)

        return y
