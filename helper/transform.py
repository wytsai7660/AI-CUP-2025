from abc import ABC, abstractmethod
from typing import List

import torch


class Transform(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass


class Compose(Transform):
    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            x = transform(x)
        return x


class Normalize(Transform):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class NormalizeSingle(Transform):
    """
    This one is weirdly good for the "gender" field.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - x.mean(dim=0)) / x.std(dim=0)


import random
class RandomExtendSegment(Transform):
    """
    隨機延長每個 segment 的結束點。

    max_frac: float
        最大延長比例範圍 [0, max_frac]
    """
    def __init__(self, max_frac: float = 0.5):
        super().__init__()
        assert 0.0 <= max_frac <= 1.0, "max_frac must be between 0 and 1"
        self.max_frac = max_frac

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x shape = (L, D)
        L = x.size(0)
        # 隨機延長比例
        frac = random.uniform(0, self.max_frac)
        extra = int(L * frac)
        new_len = min(L + extra, x.size(0))
        return x[:new_len]