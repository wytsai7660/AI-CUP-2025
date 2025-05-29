import torch
from sklearn.preprocessing import OneHotEncoder

from config import POSSIBLE_VALUES

_ENCODER = OneHotEncoder(categories=POSSIBLE_VALUES, sparse_output=False)
_ENCODER.fit(torch.tensor([[2, 2, 2, 2]]))


def labels_to_onehot(labels: torch.Tensor) -> torch.Tensor:
    if labels.dim() == 1:
        labels = labels.unsqueeze(0)
        onehot = torch.from_numpy(_ENCODER.transform(labels)).float()
        return onehot[0]
    onehot = torch.from_numpy(_ENCODER.transform(labels)).float()
    return onehot


_FIELD_LENS = [len(pv) for pv in POSSIBLE_VALUES]


def softmax(x: torch.Tensor, t: float = 1) -> torch.Tensor:
    ndim = x.ndim
    if ndim == 1:
        x = x.unsqueeze(0)
    values = torch.split(x, _FIELD_LENS, dim=-1)
    result = torch.cat([torch.softmax(value / t, dim=-1) for value in values], dim=-1)
    return result.squeeze(0) if ndim == 1 else result


if __name__ == "__main__":
    label = torch.tensor([1, 1, 2, 3])
    labels = torch.tensor([[1, 1, 2, 3], [2, 1, 0, 2]])
    onehot = labels_to_onehot(label)
    onehots = labels_to_onehot(labels)
    print(onehot)
    print(onehots)

    from math import isclose

    sample_output = torch.randn(5, 11)
    print(sample_output)
    ppp = softmax(sample_output)
    print(ppp)
    all_rows_true = all(
        all(
            isclose(p.sum().item(), 1.0, rel_tol=1e-6)
            for p in torch.split(pp, _FIELD_LENS, dim=-1)
        )
        for pp in ppp
    )
    print(all_rows_true)
