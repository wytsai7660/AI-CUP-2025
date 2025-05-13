from math import inf
from pathlib import Path
from typing import Callable, List, Tuple

import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, Dataset

from config import FIELDS, POSSIBLE_VALUES
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
        self.seg_args = dict(
            smooth_w=smooth_w, perc=perc, dist_frac=dist_frac
        )  # TODO: isolate the segmenting logic

        df_to_encode = pd.read_csv(info_csv)[FIELDS]
        encoder = OneHotEncoder(categories=POSSIBLE_VALUES, sparse_output=False)
        metas = torch.tensor(encoder.fit_transform(df_to_encode), dtype=torch.float32)

        # print(metas.shape)
        # print(metas)

        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for fpath, meta in zip(sorted(data_dir.iterdir()), metas):
            data = torch.tensor(
                pd.read_csv(fpath, sep=r"\s+", header=None).values, dtype=torch.float32
            )
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
