import numpy as np
from .IAugmentation import ISampleAugmentation


class MixinAugmentation(ISampleAugmentation):
    def __init__(self, random_seed: int = 42, threshold: float = .5):
        self.rng = np.random.RandomState(random_seed)
        self.threshold = threshold

    def transform(self, x):
        raise NotImplementedError

    def _active_augmentation(self) -> bool:
        if self.rng.uniform(low=0.0, high=1.0) <= self.threshold:
            return True

        return False

    def _get_mask(self, mask_size: int, max_mask: int):
        """True: keep words"""
        _max_mask = self.rng.randint(0, high=max_mask)
        if _max_mask == 0:
            return [True] * mask_size

        sequence = [True] * (mask_size - max_mask) + [False] * max_mask
        self.rng.shuffle(sequence)
        return sequence
