from pathlib import Path
from typing import Callable, List, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, Dataset

from config import NUM_WORKERS, POSSIBLE_VALUES, PREDICTING_FIELDS
from helper.cut import segment_file
from helper.cut_methods import *


class TrajectoryDataset(Dataset):
    """
    每筆 item 回傳 (segment, meta)，皆為 torch.Tensor：
    - segment: shape=(L, 6), dtype=torch.float32
    - meta   : one-hot vector for [gender, hand, years, level], dtype=torch.float32

    Note:
    - dataframe: should contain columns `['unique_id', 'player_id'] + PREDICTING_FIELDS`
    """

    REQUIRED_FIELDS = ["unique_id", "player_id"] + PREDICTING_FIELDS
    INF = int(1e18)

    def __init__(
        self,
        data_dir: Path,
        dataframe: pd.DataFrame,
        smooth_w: int = 5,
        perc: int = 75,
        dist_frac: float = 0.3,
        min_duration: int = -INF,
        max_duration: int = INF,
        cut_method: Callable[..., tuple[pd.DataFrame, list]] = cut_by_default(),
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self.transform = transform
        self.seg_args = dict(
            smooth_w=smooth_w, perc=perc, dist_frac=dist_frac
        )  # TODO: isolate the segmenting logic

        df_to_encode = dataframe[PREDICTING_FIELDS]
        encoder = OneHotEncoder(categories=POSSIBLE_VALUES, sparse_output=False)
        metas = torch.tensor(encoder.fit_transform(df_to_encode), dtype=torch.float32)

        # print(metas.shape)
        # print(metas)

        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        fpaths = [
            data_dir / f"{unique_id}.txt" for unique_id in dataframe["unique_id"].values
        ]
        for fpath, meta in zip(fpaths, metas):
            # print(fpath)
            data = torch.tensor(
                pd.read_csv(fpath, sep=r"\s+", header=None).values, dtype=torch.float32
            )
            # FIXME: cut_method seems to have some bug, temporarily disable it and use segment_file directly
            # TODO: uncomment the following line and remove the next line when cut_method is fixed
            # _, segs = cut_method(fpath)
            _, segs = segment_file(fpath, **self.seg_args)
            for st, ed in segs:
                duration = ed - st + 1
                if duration < min_duration or duration > max_duration:
                    continue
                self.samples.append((data[st : ed + 1, :], meta))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        if self.transform:
            return (self.transform(item[0]), item[1])
        return item


def collate_fn_torch(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
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
) -> Tuple[DataLoader, DataLoader]:
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

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        A tuple containing the training DataLoader and the validation DataLoader.

    Notes
    -----
    - The split is performed such that no `player_id` appears in both the training and validation sets.
    - Stratified splitting is used based on the `split_target` column to maintain class distribution.
    """

    df = pd.read_csv(info_csv)[TrajectoryDataset.REQUIRED_FIELDS]
    unique_player = df.drop_duplicates(subset=["player_id"])

    train_player_ids, valid_player_ids = train_test_split(
        unique_player["player_id"].to_numpy(),
        test_size=0.2,
        # random_state=42,
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
        data_dir, df[df["player_id"].isin(train_player_ids)]
    )
    valid_dataset = TrajectoryDataset(
        data_dir, df[df["player_id"].isin(valid_player_ids)]
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

    from config import TRAIN_DATA_DIR, TRAIN_INFO

    train_loader, valid_loader = get_train_valid_dataloader(
        TRAIN_DATA_DIR, TRAIN_INFO, split_target="gender", batch_size=1
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
        break
