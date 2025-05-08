import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from cut import segment_file  # 你原本提供的切段函式

class SwingDataset(Dataset):
    """
    每筆 item 回傳 (segment, meta)，全用 NumPy：
      - segment: shape=(L, 6)
      - meta   : one-hot vector for [gender, hand, years, level]
    """
    def __init__(self,
                 data_dir: str,
                 info_csv: str,
                 smooth_w: int = 5,
                 perc: int = 75,
                 dist_frac: float = 0.3,
                 min_len: int = 10,
                 max_len: int = None,
                 transform=None):
        # 基本參數
        self.data_dir = data_dir
        self.transform = transform
        self.seg_args = dict(smooth_w=smooth_w, perc=perc, dist_frac=dist_frac)

        # 1) 讀 metadata
        df = pd.read_csv(info_csv)
        # 2) 建立每個欄位的分類清單（sorted for consistency）
        self.genders = sorted(df['gender'].unique())
        self.hands   = sorted(df['hold racket handed'].unique())
        self.years   = sorted(df['play years'].unique())
        self.levels  = sorted(df['level'].unique())
        # 3) 預先為每個 unique_id 做好 one-hot vector
        self.meta_dict = {}
        for _, row in df.iterrows():
            uid = int(row['unique_id'])
            # gender
            g = np.zeros(len(self.genders), dtype=np.float32)
            g[self.genders.index(row['gender'])] = 1
            # hand
            h = np.zeros(len(self.hands), dtype=np.float32)
            h[self.hands.index(row['hold racket handed'])] = 1
            # play years
            y = np.zeros(len(self.years), dtype=np.float32)
            y[self.years.index(row['play years'])] = 1
            # level
            l = np.zeros(len(self.levels), dtype=np.float32)
            l[self.levels.index(row['level'])] = 1
            # concat
            self.meta_dict[uid] = np.concatenate([g, h, y, l], axis=0)

        # 4) 遍歷所有 .txt 檔，呼叫 segment_file，再過濾長度
        self.samples = []  # list of (file_path, start_idx, end_idx, unique_id)
        for fname in sorted(os.listdir(data_dir),
                            key=lambda f: int(os.path.splitext(f)[0])):
            if not fname.endswith('.txt'):
                continue
            uid = int(os.path.splitext(fname)[0])
            path = os.path.join(data_dir, fname)
            _, segs = segment_file(path, **self.seg_args)
            for st, ed in segs:
                L = ed - st + 1
                if L < min_len: 
                    continue
                if max_len is not None and L > max_len:
                    continue
                self.samples.append((path, st, ed, uid))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, st, ed, uid = self.samples[idx]
        data = np.loadtxt(path)          # shape = (T, 6)
        seg  = data[st:ed+1, :]          # shape = (L, 6)
        if self.transform:
            seg = self.transform(seg)
        meta = self.meta_dict[uid]       # shape = (G+H+Y+L,)
        return seg.astype(np.float32), meta

def collate_fn_numpy(batch):
    """
    batch: list of (segment (L_i,6), meta (M,))
    回傳三個 NumPy 陣列：
      padded: shape (B, L_max, 6)
      lengths: shape (B,) 實際長度
      metas:   shape (B, M)
    """
    segs, metas = zip(*batch)
    lengths = np.array([s.shape[0] for s in segs], dtype=np.int64)
    L_max = lengths.max()
    B = len(segs)
    # pad segments
    padded = np.zeros((B, L_max, 6), dtype=np.float32)
    for i, s in enumerate(segs):
        padded[i, : s.shape[0], :] = s
    metas = np.stack(metas, axis=0).astype(np.float32)
    return padded, lengths, metas

# ------------------------
# 使用範例
# ------------------------
if __name__ == '__main__':
    DATA_DIR = '/Users/yue/AI-CUP-2025/alan/data/train/train_data'
    INFO_CSV = '/Users/yue/AI-CUP-2025/alan/data/train/train_info.csv'

    ds = SwingDataset(
        data_dir=DATA_DIR,
        info_csv=INFO_CSV,
        smooth_w=5,
        perc=75,
        dist_frac=0.3,
        min_len=20,
        max_len=500,
        transform=lambda x: (x - x.mean(0)) / (x.std(0) + 1e-6)
    )
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_numpy
    )

    # Visualize 一筆 sample
    for padded, lengths, metas in loader:
        seg, length, meta = padded[0], lengths[0], metas[0]
        T = length
        t = np.arange(T) / 85.0
        channels = ['ax','ay','az','gx','gy','gz']

        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(6, 1, figsize=(10,12), sharex=True)
        for i, ch in enumerate(channels):
            axs[i].plot(t, seg[:T, i])
            axs[i].set_ylabel(ch)
            axs[i].grid(True)
        axs[-1].set_xlabel('Time (s)')
        fig.suptitle(f'Sample UID={int(meta.shape[0]>0)} — Meta sum={meta.sum():.0f}')
        plt.tight_layout()
        plt.show()
        break
