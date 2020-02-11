import numpy as np
from .BaseAugmentation import MixinAugmentation


class _BaseLabel(MixinAugmentation):
    def __init__(self, min_value: float = .05, max_value: float = .95, random_seed: int = 42, threshold: float = .5):
        super().__init__(random_seed=random_seed, threshold=threshold)
        self.min_value: float = min_value
        self.max_value: float = max_value
        if self.max_value < self.min_value:
            raise ValueError()

    def _do_scale(self, x: np.array) -> np.array:
        raise NotImplementedError()

    def transform(self, x: np.array) -> np.array:
        if self._active_augmentation:
            return self._do_scale(x)

        return x


class ClipLabel(_BaseLabel):
    def __init__(self, min_value: float = .05, max_value: float = .95, random_seed: int = 42, threshold: float = .5):
        super().__init__(min_value=min_value, max_value=max_value, random_seed=random_seed, threshold=threshold)

    def _do_scale(self, x: np.array) -> np.array:
        return np.clip(x, a_min=self.min_value, a_max=self.max_value)


class LabelSoften(_BaseLabel):
    def __init__(self, min_value: float = .05, max_value: float = .95, random_seed: int = 42, threshold: float = .5):
        super().__init__(min_value=min_value, max_value=max_value, random_seed=random_seed, threshold=threshold)
        self.intercept: float = self.max_value - self.min_value

    def _do_scale(self, x: np.array) -> np.array:
        return x * self.intercept + self.min_value
