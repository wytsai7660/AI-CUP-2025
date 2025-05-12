from math import inf
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from helper.cut import segment_file


class TrajectoryDataset(Dataset):
    """
    每筆 item 回傳 (segment, meta)，皆為 torch.Tensor：
      - segment: shape=(L, 6), dtype=torch.float32
      - meta   : one-hot vector for [gender, hand, years, level], dtype=torch.float32
    """

    def __init__(
        self,
        data_dir: Path,
        info_csv: Path,
        smooth_w: int = 5,
        perc: int = 75,
        dist_frac: float = 0.3,
        min_duration: int = -inf,
        max_duration: int = inf,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self.transform = transform
        self.seg_args = dict(smooth_w=smooth_w, perc=perc, dist_frac=dist_frac)

        df = pd.read_csv(info_csv)

        self.genders = sorted(df["gender"].unique())
        self.hands = sorted(df["hold racket handed"].unique())
        self.years = sorted(df["play years"].unique())
        self.levels = sorted(df["level"].unique())

        self.meta_dict = {}
        for _, row in df.iterrows():
            uid = int(row["unique_id"])
            g = torch.zeros(len(self.genders), dtype=torch.float32)
            g[self.genders.index(row["gender"])] = 1
            h = torch.zeros(len(self.hands), dtype=torch.float32)
            h[self.hands.index(row["hold racket handed"])] = 1
            y = torch.zeros(len(self.years), dtype=torch.float32)
            y[self.years.index(row["play years"])] = 1
            l = torch.zeros(len(self.levels), dtype=torch.float32)
            l[self.levels.index(row["level"])] = 1
            self.meta_dict[uid] = torch.cat([g, h, y, l], dim=0)

        self.samples = []  # list of (file_path, start_idx, end_idx, unique_id)
        for fpath in sorted(data_dir.iterdir()):
            # print("processing:", fpath)
            uid = int(fpath.stem)  # FIXME: to be removed?
            _, segs = segment_file(fpath, **self.seg_args)
            for st, ed in segs:
                duration = ed - st + 1
                if duration < min_duration or duration > max_duration:
                    continue
                self.samples.append((fpath, st, ed, uid))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, st, ed, uid = self.samples[idx]
        data = torch.tensor(
            pd.read_csv(path, sep=r"\s+", header=None).values, dtype=torch.float32
        )  # shape = (T, 6)
        segment = data[st : ed + 1, :]  # (L, 6)
        if self.transform:
            segment = self.transform(segment)
        meta = self.meta_dict[uid]  # (G+H+Y+L,)
        return segment, meta


def collate_fn_torch(batch):
    """
    batch: list of (segment (L_i,6), meta (M,))
    回傳：
      - padded:  Tensor shape (B, L_max, 6), float32
      - lengths: Tensor shape (B,),    int64
      - metas:   Tensor shape (B, M),  float32
    """
    segments, metas = zip(*batch)
    lengths = torch.tensor([s.size(0) for s in segments], dtype=torch.long)
    padded = torch.nn.utils.rnn.pad_sequence(segments, batch_first=True)
    metas = torch.stack(metas, dim=0)
    return padded, lengths, metas


if __name__ == "__main__":
    # Example: Visualize a training sample
    import matplotlib.pyplot as plt

    from config import TRAIN_DATA_DIR, TRAIN_INFO

    dataset = TrajectoryDataset(
        data_dir=TRAIN_DATA_DIR,
        info_csv=TRAIN_INFO,
        smooth_w=5,
        perc=75,
        dist_frac=0.3,
        min_duration=20,
        max_duration=500,
        transform=lambda x: (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6),
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn_torch
    )

    for padded, lengths, metas in loader:
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
