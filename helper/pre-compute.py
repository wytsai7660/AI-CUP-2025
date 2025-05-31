# NOTE: This script should be run only once to pre-compute some constants


from pathlib import Path
from typing import Tuple

import pandas as pd
import torch

from config import PREDICTING_FIELDS, TRAIN_DATA_DIR, TRAIN_INFO
from helper.segment import Trim


def compute_mean_std(data_dir: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the mean and std of all segments in the dataset.
    Returns:
        mean: Tensor of shape (6,)
        std:  Tensor of shape (6,)
    """

    all_data = []
    trim = Trim()
    for fpath in data_dir.iterdir():
        data = torch.tensor(
            pd.read_csv(fpath, sep=r"\s+", header=None).values, dtype=torch.float32
        )
        trimmed = trim(data)[0]
        all_data.append(trimmed)
    all_data = torch.cat(all_data, dim=0)
    print(all_data.shape)
    mean = all_data.mean(dim=0)
    std = all_data.std(dim=0)
    return mean, std


def count(info_csv: Path):
    """
    Count the number of segments in the dataset.
    """
    df = pd.read_csv(info_csv)
    df.value_counts()
    for field in PREDICTING_FIELDS:
        print(df[field].value_counts().sort_index().values)


mean, std = compute_mean_std(TRAIN_DATA_DIR)
print(f"mean: {mean}")
print(f"std : {std}")
count(TRAIN_INFO)
