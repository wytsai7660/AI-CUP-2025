# helper/csv_dataloader.py
import ast
from pathlib import Path
from typing import Tuple
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from helper.dataloader import collate_fn_torch, INF
from helper.functions import labels_to_onehot
from helper.segment import Yungan
from config import PREDICTING_FIELDS, NUM_WORKERS, SEED

def parse_cut_points(s: str) -> list[tuple[int,int,int]]:
    """
    從像 '[(55, 47, 73), (114, 106, 130), …]' 的字串中
    提取所有三元 tuple，回傳 List[(int,int,int)]。
    """
    triples = re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", s)
    return [(int(a), int(b), int(c)) for a,b,c in triples]


class CSVSegmentDataset(Dataset):
    """
    用外部 CSV（unique_id, cut point 列表）做切分，
    且不足 threshold 時 fallback 回 Yungan 分割。
    每筆 item 回傳 (segment, meta) 皆為 torch.Tensor。
    """
class CSVSegmentDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        info_df: pd.DataFrame,
        cut_csv: Path,
        threshold: int = 20,
        min_duration: int = -INF,
        max_duration: int = INF,
        transform=None,
    ):
        # 1. Meta labels → one-hot
        labels = torch.from_numpy(info_df[PREDICTING_FIELDS].values)
        metas = labels_to_onehot(labels)  # shape = (N, M)

        # 2. 讀 CSV 並建立 unique_id → cut_list 映射
        cut_df = pd.read_csv(cut_csv, dtype={"cut point": str})
        self.cut_map = {
            int(row["unique_id"]): parse_cut_points(row["cut point"])
            for _, row in cut_df.iterrows()
        }

        self.samples: list[Tuple[torch.Tensor, torch.Tensor]] = []
        for uid, meta in zip(info_df["unique_id"].values, metas):
            fpath = data_dir / f"{uid}.txt"
            data = torch.from_numpy(np.loadtxt(fpath)).float()

            # 3. 依 CSV 切分或 Yungan fallback
            raw_cuts = self.cut_map.get(int(uid), [])
            if len(raw_cuts) >= threshold:
                segs = []
                for peak_idx, trough_b, trough_a in raw_cuts:
                    # 以 trough_b → trough_a 作為 segment
                    seg = data[trough_b:trough_a]
                    segs.append(seg)
            else:
                segs = Yungan()(data)

            # 4. 過濾長度並蒐集
            for seg in segs:
                L = seg.shape[0]
                if min_duration <= L <= max_duration:
                    if transform:
                        seg = transform(seg)
                    self.samples.append((seg, meta))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def get_csv_train_valid_dataloader(
    data_dir: Path,
    info_csv: Path,
    cut_csv: Path,
    threshold: int = 20,
    split_target: str = "gender",
    batch_size: int = 32,
    min_duration: int = -INF,
    max_duration: int = INF,
    train_transform=None,
    valid_transform=None,
) -> Tuple[DataLoader, DataLoader]:
    """
    與 get_train_valid_dataloader 等價，但 segmentation 換成 CSV + fallback。

    1. 讀取 info_csv，按 player_id 分組 stratify by split_target
    2. 建立 CSVSegmentDataset 的 train/valid
    3. 用 collate_fn_torch 回傳 DataLoader

    使用範例：
        from config import TRAIN_DATA_DIR, TRAIN_INFO
        from helper.transform import Compose, RandomExtendSegment
        from helper.csv_dataloader import get_csv_train_valid_dataloader
        PROJECT_ROOT = Path(__file__).resolve().parent.parent
        cut_csv_path = PROJECT_ROOT / 'helper' / 'hmm_cut_points_results.csv'
        train_loader, valid_loader = get_csv_train_valid_dataloader(
            data_dir=TRAIN_DATA_DIR,
            info_csv=TRAIN_INFO,
            cut_csv=cut_csv_path,
            threshold=20,
            split_target=PREDICTING_FIELDS[0],  # e.g. 'gender'
            batch_size=32,
            train_transform=Compose([RandomExtendSegment(max_frac=0.5)]),
            valid_transform=None
        )
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Valid samples: {len(valid_loader.dataset)}")
    """
    # 1. 讀 meta 資訊、按 player_id 切分 train/valid
    df = pd.read_csv(info_csv)
    unique_player = df.drop_duplicates(subset=["player_id"])
    train_pids, valid_pids = __import__("sklearn.model_selection").model_selection.train_test_split(  # noqa
        unique_player["player_id"].to_numpy(),
        test_size=0.2,
        random_state=SEED,
        stratify=unique_player[split_target].to_numpy(),
    )
    train_df = df[df["player_id"].isin(train_pids)]
    valid_df = df[df["player_id"].isin(valid_pids)]

    # 2. 建立 Dataset
    train_ds = CSVSegmentDataset(
        data_dir,
        train_df,
        cut_csv,
        threshold=threshold,
        min_duration=min_duration,
        max_duration=max_duration,
        transform=train_transform,
    )
    valid_ds = CSVSegmentDataset(
        data_dir,
        valid_df,
        cut_csv,
        threshold=threshold,
        min_duration=min_duration,
        max_duration=max_duration,
        transform=valid_transform,
    )

    # 3. DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn_torch,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn_torch,
    )
    return train_loader, valid_loader
