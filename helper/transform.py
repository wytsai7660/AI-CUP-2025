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
