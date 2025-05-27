from pathlib import Path
from typing import final, override

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import MEAN, NUM_WORKERS, PREDICTING_FIELDS, SEED
from helper.functions import labels_to_onehot
from helper.segment import Segment
from helper.transform import Transform

INF = int(1e18)


@final
class TrajectoryDataset(Dataset):
    """
    每筆 item 回傳 (segment, meta)，皆為 torch.Tensor:
    - seg  : shape=(L, 6), dtype=torch.float32
    - meta : one-hot vector for [gender, hand, years, level], dtype=torch.float32
    """

    def __init__(
        self,
        data_dir: Path,
        dataframe: pd.DataFrame,
        min_duration: int = -INF,
        max_duration: int = INF,
        segment: Segment | None = None,
        transform: Transform | None = None,
    ):
        self.transform = transform

        labels = torch.from_numpy(dataframe[PREDICTING_FIELDS].values)
        metas = labels_to_onehot(labels)
        # print(metas.shape)
        # print(metas)

        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        fpaths = [data_dir / f"{id}.txt" for id in dataframe["unique_id"].values]
        for fpath, meta in tqdm(
            zip(fpaths, metas), desc="Loading Data", total=len(fpaths)
        ):
            data = torch.from_numpy(np.loadtxt(fpath)).float()
            segs = segment(data) if segment else [data]
            self.samples += [
                (seg, meta)
                for seg in segs
                if min_duration <= seg.shape[0] <= max_duration
            ]

    def __len__(self):
        return len(self.samples)

    @override
    def __getitem__(self, idx: int):
        item = self.samples[idx]
        if self.transform:
            return (self.transform(item[0]), item[1])
        return item


def collate_fn_torch(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    """
    batch: list of (segment (L_i,6), meta (M,))
    回傳：
      - padded:  Tensor shape (B, L_max, 6), float32
      - lengths: Tensor shape (B,),    int64
      - metas:   Tensor shape (B, M),  float32
    """
    segments, metas = zip(*batch)
    lengths = torch.tensor([s.size(0) for s in segments], dtype=torch.long)
    padded = torch.nn.utils.rnn.pad_sequence(segments, batch_first=True)  # type: ignore
    metas = torch.stack(metas, dim=0)
    return padded, lengths, metas


def get_train_valid_dataloader(
    data_dir: Path,
    info_csv: Path,
    split_target: str = "gender",
    batch_size: int = 32,
    min_duration: int = -INF,
    max_duration: int = INF,
    segment: Segment | None = None,
    train_transform: Transform | None = None,
    valid_transform: Transform | None = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Splits the data into training and validation DataLoaders

    Parameters
    ----------
    `data_dir`: `Path`
        The directory containing the data files.
    `info_csv`: `Path`
        Path to the CSV file containing metadata and labels for the dataset.
    `split_target`: `str`, optional
        The column name in the CSV to use for stratified splitting, can be one of the fields in `PREDICTING_FIELDS` (default is "gender").
    `batch_size`: `int`, optional
        The batch size to use for the DataLoaders (default is 32).

    Parameters
    ------------------
    `min_duration`: `int`, optional
        The minimum duration of segments to include in the dataset (default is -INF).
    `max_duration`: `int`, optional
        The maximum duration of segments to include in the dataset (default is INF).
    `segment`: `Segment` | `None`, optional
        The segmenting method to apply to the data (default is None, which means no segmentation).
    `train_transform`: `Transform` | `None`, optional
        The transformation to apply to the training data (default is None, which means no transformation).
    `valid_transform`: `Transform` | `None`, optional
        The transformation to apply to the validation data (default is None, which means no transformation).


    Returns
    -------
    Tuple[DataLoader, DataLoader]
        A tuple containing the training DataLoader and the validation DataLoader.

    Notes
    -----
    - The split is performed such that no `player_id` appears in both the training and validation sets.
    - Stratified splitting is used based on the `split_target` column to maintain class distribution.
    """

    df = pd.read_csv(info_csv)
    unique_player = df.drop_duplicates(subset=["player_id"])

    train_player_ids, valid_player_ids = train_test_split(
        unique_player["player_id"].to_numpy(),
        test_size=0.2,
        random_state=SEED,
        stratify=unique_player[split_target].to_numpy(),
    )

    # from collections import defaultdict

    # print(f"train player ids: {train_player_ids}")
    # print(f"test player ids: {test_player_ids}")

    # count = defaultdict(int)
    # for i in train_player_ids:
    #     tmp = int(df[df["player_id"] == i][split_target].values[0])
    #     # print(tmp)
    #     count[tmp] += 1

    # print(f"train count: {dict(count)}")

    # count = defaultdict(int)
    # for i in test_player_ids:
    #     tmp = int(df[df["player_id"] == i][split_target].values[0])
    #     # print(tmp)
    #     count[tmp] += 1

    # print(f"test count: {dict(count)}")

    train_dataset = TrajectoryDataset(
        data_dir,
        df[df["player_id"].isin(train_player_ids)],
        min_duration=min_duration,
        max_duration=max_duration,
        segment=segment,
        transform=train_transform,
    )
    valid_dataset = TrajectoryDataset(
        data_dir,
        df[df["player_id"].isin(valid_player_ids)],
        min_duration=min_duration,
        max_duration=max_duration,
        segment=segment,
        transform=valid_transform,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn_torch,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn_torch,
    )

    return (train_dataloader, valid_dataloader)


if __name__ == "__main__":
    # Example: Visualize a training sample
    import matplotlib.pyplot as plt

    from config import MEAN, STD, TRAIN_DATA_DIR, TRAIN_INFO

    # Use * here just for quick testing
    from helper.segment import *
    from helper.transform import *

    train_loader, valid_loader = get_train_valid_dataloader(
        TRAIN_DATA_DIR,
        TRAIN_INFO,
        split_target="gender",
        batch_size=1,
        segment=Yungan(),
        train_transform=Compose(
            [
                Normalize(mean=MEAN, std=STD),
            ]
        ),
        valid_transform=Compose(
            [
                Normalize(mean=MEAN, std=STD),
            ]
        ),
    )

    for padded, lengths, metas in train_loader:
        seg = padded[0]  # (L,6)
        length = lengths[0].item()  # L
        t = torch.arange(length).float() / 85.0
        channels = ["ax", "ay", "az", "gx", "gy", "gz"]

        fig, axs = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
        for i, ch in enumerate(channels):
            axs[i].plot(t.numpy(), seg[:length, i].numpy())
            axs[i].set_ylabel(ch)
            axs[i].grid(True)
        axs[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.show()
        print("Meta shape:", metas.shape)
        print("Meta vector:", metas[0])
        # break
