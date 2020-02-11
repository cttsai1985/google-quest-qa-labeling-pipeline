from typing import List
from itertools import compress
from .BaseAugmentation import MixinAugmentation


class _RandomTruncate(MixinAugmentation):
    def __init__(self, min_length: int = 128, max_length: int = 256, random_seed: int = 42, threshold: float = .5):
        super().__init__(random_seed=random_seed, threshold=threshold)
        self.min_length: int = min_length
        self.max_length: int = max_length

    def _do_truncate(self, x, length: int):
        raise NotImplementedError

    def transform(self, x: List[str]) -> List[str]:
        len_x: int = len(x)
        if len_x <= self.min_length:
            return x

        if self._active_augmentation:
            seq = (self.min_length, min(len_x, self.max_length))
            length = self.rng.randint(min(seq), max(seq))
            return self._do_truncate(x, length)

        return x


class RandomTruncateHead(_RandomTruncate):
    def __init__(self, min_length: int = 128, max_length: int = 256, random_seed: int = 42, threshold: float = .5):
        super().__init__(min_length=min_length, max_length=max_length, random_seed=random_seed, threshold=threshold)

    def _do_truncate(self, x, length: int):
        return x[-length:]


class RandomTruncateTail(_RandomTruncate):
    def __init__(self, min_length: int = 128, max_length: int = 256, random_seed: int = 42, threshold: float = .5):
        super().__init__(min_length=min_length, max_length=max_length, random_seed=random_seed, threshold=threshold)

    def _do_truncate(self, x, length: int):
        return x[:length]


class RandomDropWords(MixinAugmentation):
    def __init__(
            self, min_length: int = 1, max_drop: int = 5, drop_rate: float = .1, random_seed: int = 42,
            threshold: float = .5):
        super().__init__(random_seed=random_seed, threshold=threshold)
        self.min_length: int = min_length
        self.max_drop: int = max_drop
        self.drop_rate: float = drop_rate

    def transform(self, x: List[str]) -> List[str]:
        len_x: int = len(x)
        if len_x < self.min_length:
            return x

        if self._active_augmentation:
            max_drop = min(max(0, len_x - self.min_length), max(int(self.drop_rate * len_x), self.max_drop))
            if max_drop < 1:
                return x

            mask = self._get_mask(len_x, max_drop)
            x = list(compress(x, mask))

        return x

