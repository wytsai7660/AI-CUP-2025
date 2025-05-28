import re
import warnings
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import List, final

import numpy as np
import ruptures as rpt
import torch
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, detrend, filtfilt, find_peaks, lfilter
from sklearn.preprocessing import MinMaxScaler

"""
The segmenting method to apply to the data
----------
`Segment`: base class
`Trim`: 僅去除頭尾無波動部分，不進行分段。
`FixedSize`: 去除頭尾無波動部分，切出為固定長度的段。
`Yungan`: 重新偵測波動，切出不定長度的段。
`ChangePoint`: 利用 ruptures 檢測變化點，切出不定長度的段。
"""


class Segment(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, data: torch.Tensor) -> List[torch.Tensor]:
        pass


class Trim(Segment):
    """
    僅去除頭尾無波動部分，不進行分段。

    Params
    ------
    smooth_window : int
        平滑窗口大小（样本数）
    percentage : float
        能量阈值的百分位数 (0-100)
    """

    def __init__(self, smooth_window: int = 5, percentage: float = 75):
        self.smooth_window = smooth_window
        self.percentage = percentage

    def __call__(self, data: torch.Tensor) -> List[torch.Tensor]:
        # Consider only the acceleration vector (ax, ay, az)
        magnitude = torch.linalg.vector_norm(data[:, :3], dim=1)
        magnitude = uniform_filter1d(magnitude, size=self.smooth_window)

        threshold = np.percentile(magnitude, self.percentage)
        active = magnitude > threshold
        first = np.argmax(active)
        last = len(active) - 1 - np.argmax(active[::-1])

        return [data[first:last]]


class FixedSize(Trim):
    """
    去除頭尾無波動部分，切出為固定長度的段。

    Note
    ----
    If the tail is not fit to size, just discard it.
    """

    def __init__(self, smooth_window: int = 5, percentage: float = 75, size: int = 20):
        super().__init__(smooth_window, percentage)
        self.size = size

    def __call__(self, data: torch.Tensor) -> List[torch.Tensor]:
        trimmed = super().__call__(data)[0]
        segs, start = [], 0
        while start + self.size <= len(trimmed):
            segs.append(trimmed[start : start + self.size])
            start += self.size
        return segs


class Yungan(Segment):
    """
    重新偵測波動，切出不定長度的段。

    Note
    ----
    維元看不懂，所以這部分是直接挪用原 `helper/cut.py` 中的實現。
    """

    def __init__(
        self,
        smooth_w=5,
        perc=75,
        dist_frac=0.3,
    ):
        self.smooth_w = smooth_w
        self.perc = perc
        self.dist_frac = dist_frac

    def __call__(self, data: torch.Tensor) -> List[torch.Tensor]:
        def estimate_period(a_env):
            acf = np.correlate(a_env - a_env.mean(), a_env - a_env.mean(), mode="full")
            acf = acf[len(acf) // 2 :]
            peaks, _ = find_peaks(acf, distance=1)
            return peaks[1] if len(peaks) > 1 else len(a_env) // 10

        acc = data[:, :3]
        a_env = uniform_filter1d(np.linalg.norm(acc, axis=1), size=self.smooth_w)
        T = estimate_period(a_env)
        height = np.percentile(a_env, self.perc)
        distance = max(int(T * self.dist_frac), 1)
        peaks, _ = find_peaks(a_env, height=height, distance=distance)
        boundaries = [
            np.argmin(a_env[peaks[i] : peaks[i + 1]]) + peaks[i]
            for i in range(len(peaks) - 1)
        ]
        segs, start = [], 0
        for b in boundaries:
            segs.append(data[start:b])
            start = b
        segs.append(data[start : len(a_env) - 1])
        return segs


class ChangePoint(Segment):
    """
    利用 ruptures 檢測變化點，切出不定長度的段。

    Params
    ------
    file_path : str
        IMU 数据文件路径，6 列无表头：ax,ay,az,gx,gy,gz
    smooth_w : int
        对加速度模长做均值滤波的窗口大小
    pen : float or int
        ruptures.Pelt 的 penalty 参数，越小分段越多
    model : str
        ruptures 使用的代价模型，如 "rbf", "l2" 等

    Returns
    -------
    df : pandas.DataFrame
        原始 6 轴数据
    segs : List[Tuple[int,int]]
        切分出的各段 (start_idx, end_idx)
    """

    def __init__(self, smooth_w=5, pen: float | int = 3, model="rbf"):

        self.smooth_w = smooth_w
        self.pen = pen
        self.model = model
        pass

    def __call__(self, data: torch.Tensor) -> List[torch.Tensor]:

        acc = data[:, :3]

        a_env = uniform_filter1d(np.linalg.norm(acc, axis=1), size=self.smooth_w)
        # 3. 用 PELT 检测变化点
        algo = rpt.Pelt(model=self.model).fit(a_env)
        # bkps 包含最后一个点 len(a_env)
        bkps = algo.predict(pen=self.pen)
        # 去掉末尾，不当作真实边界
        boundaries = bkps[:-1]
        # 4. 构造各段区间
        segs = []
        start = 0
        for b in boundaries:
            segs.append(data[start:b])
            start = b
        # 最后一段
        segs.append(data[start : len(a_env) - 1])
        return segs


class IntergratedSplit(Segment):
    """
    實現 https://arxiv.org/abs/2306.17550 中提及的 Waveform Split Algorithm

    如果偵測到的揮拍次數不夠，則使用Yungan的分段方法
    """

    def __init__(self):
        pass

    @staticmethod
    def lowpass_filter(data, cutoff=5, fs=85, order=4):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        return filtfilt(b, a, data)

    @staticmethod
    def find_k(waveform, crossings_num=54, step=0.001):
        max_y = 1.0
        min_y = 0.0
        k = max_y

        cnt = 0
        while k >= min_y:
            shifted = waveform - k
            signs = np.sign(shifted)
            crossings = np.where(np.diff(signs))[0]
            n_crossings = len(crossings)

            if n_crossings >= crossings_num:
                # if cnt < 5:
                #     cnt += 1
                # else:
                return k, crossings
            k -= step

        return None, None

    @staticmethod
    def find_troughs(waveform, crossing_indices):
        assert (
            len(crossing_indices) % 2 == 0
        ), f"Number of crossings must be even, got {len(crossing_indices)}"

        # make direction alternate between -1 and 1
        direction = np.tile([-1, 1], len(crossing_indices) // 2 + 1)[
            : len(crossing_indices)
        ]

        trough_indices = crossing_indices.copy()
        searching = np.ones_like(trough_indices, dtype=bool)

        while np.any(searching):
            next_indices = np.clip(trough_indices + direction, 0, len(waveform) - 1)
            move_mask = (waveform[next_indices] < waveform[trough_indices]) & searching
            trough_indices[move_mask] = next_indices[move_mask]
            searching &= move_mask

        return trough_indices

    def __call__(self, data: torch.Tensor) -> List[torch.Tensor]:
        data_np = data.numpy().copy()
        integrated_waveform = np.sum(np.abs(data_np), axis=1)
        detrend_waveform = detrend(integrated_waveform)
        filtered_waveform = self.lowpass_filter(detrend_waveform)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_waveform = scaler.fit_transform(
            filtered_waveform.reshape(-1, 1)
        ).flatten()
        k, crossings = self.find_k(scaled_waveform)

        if k is None:
            warnings.warn(
                "IntegratedSplit: fallback to Yungan method due to insufficient swings",
                UserWarning,
            )
            # fallback to yungan's method
            return Yungan()(data)

        trough_indices = self.find_troughs(scaled_waveform, crossings)

        segs = [
            (s.item(), e.item())
            for s, e in zip(trough_indices[::2], trough_indices[1::2])
        ]

        return segs

class HMM(Segment):
    
    def __init__(
        self,
        n_state: int = 4,
        random_state: int = 42,
        freq_cutoff: float = 30,
        sample_rate: float = 85,
    ):
        from hmmlearn import hmm

        self.n_state = n_state
        self.random_state = random_state
        self.freq_cutoff = freq_cutoff
        self.sample_rate = sample_rate
        
    def analyze_state_segments(self, state_seq, data_seq):
        state_seq = np.array(state_seq)
        data_seq = np.array(data_seq)

        segments = []
        start = 0
        for i in range(1, len(state_seq)):
            if state_seq[i] != state_seq[i - 1]:
                segments.append((state_seq[i - 1], start, i - 1))
                start = i
        segments.append((state_seq[-1], start, len(state_seq) - 1))

        state_to_values = {}
        for state, start, end in segments:
            segment_mean = data_seq[start:end+1].mean()
            if state not in state_to_values:
                state_to_values[state] = []
            state_to_values[state].append(segment_mean)

        state_avg = {state: np.mean(values) for state, values in state_to_values.items()}
        
        best_state = max(state_avg.items(), key=lambda x: x[1])[0]

        return segments, best_state
    
    def most_common_trigram_span(self, segments, best_state):
        state_seq = [seg[0] for seg in segments]
        trigram_data = []

        for i in range(len(state_seq) - 2):
            trigram = tuple(state_seq[i:i+3])
            if trigram[1] == best_state:
                seg_triplet = segments[i:i+3]
                trigram_data.append((trigram, seg_triplet))

        if not trigram_data:
            return None, [], 0

        counter = Counter([item[0] for item in trigram_data])
        most_common_trigram, count = counter.most_common(1)[0]

        spans = []
        for trig, segs in trigram_data:
            if trig == most_common_trigram:
                start = segs[0][1]
                end = segs[2][2]
                spans.append((start, end))

        return most_common_trigram, spans, count

        
    def __call__(self, data: torch.Tensor) -> List[torch.Tensor]:
        data = detrend(data.numpy(), axis=0)
        T = len(data)
        
        normal_cutoff = self.freq_cutoff / (0.5 * self.sample_rate)
        b, a = butter(2, normal_cutoff, btype='low', analog=False)
        filtered_data = np.zeros_like(data)
        for i in range(6):
            filtered_data[:, i] = lfilter(b, a, data[:, i])
            
        acc_magnitude = np.sqrt(filtered_data[:, 0]**2 + filtered_data[:, 1]**2 + filtered_data[:, 2]**2)
        gyro_magnitude = np.sqrt(filtered_data[:, 3]**2 + filtered_data[:, 4]**2 + filtered_data[:, 5]**2)
        features = np.column_stack((filtered_data, acc_magnitude, gyro_magnitude))
        scaler = MinMaxScaler()
        features = scaler.fit_transform(features)
        
        model = hmm.GaussianHMM(
            n_components=self.n_state,
            covariance_type="full",
            random_state=self.random_state,
            n_iter=100,
        )
        model.fit(features)
        logprob, states = model.decode(features, algorithm="viterbi")
        
        segments, best_state = self.analyze_state_segments(states, acc_magnitude)
        most_common_trigram, spans, count = self.most_common_trigram_span(segments, best_state)
        
        segs = []
        for start, end in spans:
            segs.append(data[start:end])
        return segs


def parse_cut_points(s: str) -> list[tuple[int, int, int]]:
    """
    從像 '[(55, 47, 73), (114, 106, 130), …]' 的字串中
    提取所有三元 tuple，回傳 List[(int,int,int)]。
    """
    triples = re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", s)
    return [(int(a), int(b), int(c)) for a, b, c in triples]


@final
class UseCutPoints:
    """
    用預運算好的 CSV（unique_id, cut point 列表）分段（通常是因為分段方法極慢）

    Note
    ----
    這個類別不繼承 Segment，並且使用方法也不同。
    """

    def __init__(
        self, cut_points_csv: Path, threshold: int = 20, fallback: Segment = Yungan()
    ):
        self.threshold = threshold
        self.fallback = fallback

        df = pd.read_csv(cut_points_csv, dtype={"cut point": str})
        triplets = df["cut point"].apply(parse_cut_points)
        self.cut_map = dict(zip(df["unique_id"], triplets))

    def __call__(self, data: torch.Tensor, id: int) -> List[torch.Tensor]:
        triplets = self.cut_map.get(id, [])
        segs = (
            [data[start:end] for _, start, end in triplets]
            if len(triplets) >= self.threshold
            else self.fallback(data)
        )
        return segs


# class cut_by_hmm(cut_method):
#     """
#     HMM（Hidden Markov Model）：把每个时刻的传感器特征看成观测，隐藏状态分成「静止／运动」两类，利用 HMM 来解码状态序列并提取连续的运动段
#     """

#     def __init__(
#         self,
#         n_state: int = 2,
#         feature_cols: tuple[str, ...] = ("ax", "ay", "az"),
#         cov_type: str = "full",
#         random_state: int = 42,
#     ):
#         self.n_state = n_state
#         self.feature_cols = feature_cols
#         self.cov_type = cov_type
#         self.random_state = random_state

#     def __call__(self, file_path: Path) -> tuple[pd.DataFrame, list]:
#         # 1) 读取数据
#         df = pd.read_csv(
#             file_path,
#             sep=r"\s+",
#             header=None,
#             names=["ax", "ay", "az", "gx", "gy", "gz"],
#         )
#         # 2) 构建观测矩阵：这里只取加速度，也可叠加角速度
#         X = df[list(self.feature_cols)].values

#         # 3) 训练 HMM
#         model = GaussianHMM(
#             n_components=self.n_state,
#             covariance_type=self.cov_type,
#             random_state=self.random_state,
#         )
#         model.fit(X)

#         # 4) 解码状态序列
#         states = model.predict(X)  # 每个时刻属于哪个隐藏状态

#         # 5) 选取“运动”那一类（假定均值更大）
#         means = [model.means_[i].mean() for i in range(self.n_state)]
#         move_state = int(np.argmax(means))

#         # 6) 把连续的 move_state 段拼成 (start,end)
#         segs = []
#         active = states == move_state
#         idx = np.where(active)[0]
#         if idx.size:
#             # 找到断点
#             splits = np.nonzero(np.diff(idx) > 1)[0]
#             starts = [idx[0]] + [idx[i + 1] for i in splits]
#             ends = [idx[i] for i in splits] + [idx[-1]]
#             segs = [(int(s), int(e)) for s, e in zip(starts, ends)]

#         return df, segs


# class cut_by_ts_closuring(cut_method):
#     """
#     Note: should not use this method.
#     """

#     def __init__(
#         self,
#         window_size: int = 128,
#         step: int = 64,
#         n_cluster: int = 2,
#         feature_func=None,
#     ):
#         self.window_size = window_size
#         self.step = step
#         self.n_cluster = n_cluster
#         self.feature_func = feature_func
#         # 默认特征：能量均值 & 方差
#         if self.feature_func is None:
#             self.feature_funcs = [lambda w: w.mean(), lambda w: w.std()]

#     def __call__(self, file_path: Path) -> tuple[pd.DataFrame, list]:
#         df = pd.read_csv(
#             file_path,
#             sep=r"\s+",
#             header=None,
#             names=["ax", "ay", "az", "gx", "gy", "gz"],
#         )
#         # 1) 提取加速度模长
#         mag = np.linalg.norm(df[["ax", "ay", "az"]].values, axis=1)
#         N = len(mag)

#         # 2) 划小窗并提取特征
#         feats = []
#         idxs = []
#         for i in range(0, N - self.window_size + 1, self.step):
#             w = mag[i : i + self.window_size]
#             feats.append([f(w) for f in self.feature_funcs])
#             idxs.append(i)
#         feats = np.array(feats)

#         # 3) KMeans 聚成两类
#         km = KMeans(n_clusters=self.n_cluster, random_state=0).fit(feats)
#         labels = km.labels_

#         # 4) 选“运动类”：均值特征更大的那个簇
#         cluster_means = feats.mean(axis=1)
#         cls_mean = [cluster_means[labels == c].mean() for c in range(self.n_cluster)]
#         move_cls = int(np.argmax(cls_mean))

#         # 5) 把连续的“运动窗”合并成大段
#         active_windows = labels == move_cls
#         segs = []
#         current = None
#         for win_idx, is_active in enumerate(active_windows):
#             start_sample = idxs[win_idx]
#             end_sample = start_sample + self.window_size - 1
#             if is_active:
#                 if current is None:
#                     current = [start_sample, end_sample]
#                 else:
#                     # 如果与上一个窗连在一起，就扩展末尾
#                     if start_sample <= current[1] + self.step:
#                         current[1] = end_sample
#                     else:
#                         segs.append(tuple(current))
#                         current = [start_sample, end_sample]
#             else:
#                 if current is not None:
#                     segs.append(tuple(current))
#                     current = None
#         if current is not None:
#             segs.append(tuple(current))

#         return df, segs


if __name__ == "__main__":
    import argparse

    import matplotlib.pyplot as plt
    import pandas as pd

    from config import CHANNELS, TRAIN_DATA_DIR

    # Example usage:
    # 1.
    # uv run -m helper.segment -m "Yungan()"
    # 2.
    # uv run -m helper.segment -m "FixedSize(size=40)"
    # 3.
    # uv run -m helper.segment -m "lambda x: [x]"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        required=True,
        help="切分方法",
    )
    args = parser.parse_args()
    method: str = args.method

    for fpath in sorted(TRAIN_DATA_DIR.iterdir(), key=lambda x: int(x.stem)):
        data = torch.tensor(
            pd.read_csv(fpath, sep=r"\s+", header=None).values, dtype=torch.float32
        )

        plt.figure(figsize=(12, 12))
        plt.title(f"{method}, {fpath.name}")
        plt.axis("off")

        for i, channel in enumerate(CHANNELS):
            plt.subplot(len(CHANNELS), 1, i + 1)

            # HACK: use eval() to dynamically call the segment method
            segment = eval(method)
            segs = segment(data)
            segs = [seg[:, i] for seg in segs]

            x_offset = 0
            for y in segs:
                x = np.arange(len(y)) + x_offset
                plt.plot(x, y)
                x_offset += len(y)

            plt.xticks([])
            plt.ylabel(channel)

        plt.tight_layout()
        plt.show()
