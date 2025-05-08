import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from helper.cut import segment_file  # 你原本提供的切段函式

class SwingDataset(Dataset):
    """
    每筆 item 回傳 (segment, meta)，皆為 torch.Tensor：
      - segment: shape=(L, 6), dtype=torch.float32
      - meta   : one-hot vector for [gender, hand, years, level], dtype=torch.float32
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
        self.data_dir = data_dir
        self.transform = transform
        self.seg_args = dict(smooth_w=smooth_w, perc=perc, dist_frac=dist_frac)

        # 1) 讀 metadata
        df = pd.read_csv(info_csv)
        # 2) 建立每個欄位的分類清單
        self.genders = sorted(df['gender'].unique())
        self.hands   = sorted(df['hold racket handed'].unique())
        self.years   = sorted(df['play years'].unique())
        self.levels  = sorted(df['level'].unique())
        # 3) 預先為每個 unique_id 做好 one-hot tensor
        self.meta_dict = {}
        for _, row in df.iterrows():
            uid = int(row['unique_id'])
            g = torch.zeros(len(self.genders), dtype=torch.float32)
            g[self.genders.index(row['gender'])] = 1
            h = torch.zeros(len(self.hands), dtype=torch.float32)
            h[self.hands.index(row['hold racket handed'])] = 1
            y = torch.zeros(len(self.years), dtype=torch.float32)
            y[self.years.index(row['play years'])] = 1
            l = torch.zeros(len(self.levels), dtype=torch.float32)
            l[self.levels.index(row['level'])] = 1
            self.meta_dict[uid] = torch.cat([g, h, y, l], dim=0)

        # 4) 遍歷所有 .txt 檔，呼叫 segment_file，再過濾長度
        self.samples = []  # list of (file_path, start_idx, end_idx, unique_id)
        for fname in sorted(os.listdir(data_dir),
                            key=lambda f: int(os.path.splitext(f)[0])):
            if not fname.endswith('.txt'): continue
            uid = int(os.path.splitext(fname)[0])
            path = os.path.join(data_dir, fname)
            _, segs = segment_file(path, **self.seg_args)
            for st, ed in segs:
                L = ed - st + 1
                if L < min_len: continue
                if max_len is not None and L > max_len: continue
                self.samples.append((path, st, ed, uid))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, st, ed, uid = self.samples[idx]
        # 使用相對路徑從當前工作目錄讀檔
        data = torch.tensor(
            pd.read_csv(path, sep=r"\s+", header=None).values,
            dtype=torch.float32
        )  # shape = (T, 6)
        segment = data[st:ed+1, :]  # (L, 6)
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


# ------------------------
# 使用範例
# ------------------------
if __name__ == '__main__':
    PARENT   = os.path.dirname(__file__) + "/.."
    DATA_DIR = os.path.join(PARENT, 'data', 'train', 'train_data')
    INFO_CSV = os.path.join(PARENT, 'data', 'train', 'train_info.csv')

    ds = SwingDataset(
        data_dir=DATA_DIR,
        info_csv=INFO_CSV,
        smooth_w=5,
        perc=75,
        dist_frac=0.3,
        min_len=20,
        max_len=500,
        transform=lambda x: (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)
    )
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_torch
    )

    # Visualize 一筆 sample
    import matplotlib.pyplot as plt
    for padded, lengths, metas in loader:
        seg    = padded[0]         # (L,6)
        length = lengths[0].item() # L
        t = torch.arange(length).float() / 85.0
        channels = ['ax','ay','az','gx','gy','gz']

        fig, axs = plt.subplots(6, 1, figsize=(10,12), sharex=True)
        for i, ch in enumerate(channels):
            axs[i].plot(t.numpy(), seg[:length, i].numpy())
            axs[i].set_ylabel(ch)
            axs[i].grid(True)
        axs[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        plt.show()
        print("Meta shape:", metas.shape)
        print("Meta vector:", metas[0])
        break
