from typing import List


class ISampleAugmentation:
    def transform(self, x: List[str]) -> List[str]:
        raise NotImplementedError
