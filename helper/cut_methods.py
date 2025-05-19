from pathlib import Path
import pandas as pd
import numpy as np
import ruptures as rpt
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from abc import ABC, abstractmethod
import re
from config import TRAIN_DATA_DIR, TEST_DATA_DIR, TRAIN_INFO, TEST_INFO
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans

"""
Methods to 'cut' dataset
----------
`no_cut`: only cut head and tail, returns one segment.
`cut_by_default`: cut by default cut point in `xxx_info.csv`
`cut_by_segment_file`: cut by 'segment_file' method
`cut_by_fixed_size`: cut by fixed size, default size=20
`cut_by_change_point`: cut by change point of energy, slow
"""


class cut_method(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, file_path: Path) -> tuple[pd.DataFrame, list]:
        pass


class cut_by_default(cut_method):

    def __init__(self):
        pass

    def __call__(self, file_path: Path) -> tuple[pd.DataFrame, list]:
        df = pd.read_csv(
            file_path,
            sep=r"\s+",
            header=None,
            names=["ax", "ay", "az", "gx", "gy", "gz"],
        )
        csv_path = file_path.parents[1]
        if csv_path == TEST_DATA_DIR.parent:
            csv_path = TEST_INFO
        elif csv_path == TRAIN_DATA_DIR.parent:
            csv_path = TRAIN_INFO
        else:
            raise ValueError("File path should in [test, train].")
        unique_id = int(file_path.stem)
        pattern = re.compile(r"\d+")
        data = pd.read_csv(csv_path)
        item = data.groupby(by=["unique_id"]).get_group((unique_id,)).to_numpy()
        cut = np.array(list(map(int, re.findall(pattern, item[0][-1]))))
        segs, start = [], 0
        for b in cut:
            segs.append((start, b))
            start = b
        return df, segs


class no_cut(cut_method):
    """
    根据加速度模长的能量包络，掐掉头尾不活跃部分，
    并返回所有连续活跃区间的 (start, end) 列表。

    Params
    ------
    smooth_w : int
        平滑窗口大小（样本数）
    perc : float
        能量阈值的百分位数 (0–100)

    Returns
    -------
    df: pandas.DataFrame
        dataframe of raw data
    segs : List[Tuple[start,end]]
        连续活跃区间索引 [start, end).
        if no active segment, return empty list.
    """

    def __init__(self, smooth_w: int = 5, perc: float = 75):
        self.smooth_w = smooth_w
        self.perc = perc
        pass

    def __call__(self, file_path: Path) -> tuple[pd.DataFrame, list]:

        # 1. 读数据
        df = pd.read_csv(
            file_path,
            sep=r"\s+",
            header=None,
            names=["ax", "ay", "az", "gx", "gy", "gz"],
        )
        # 2. 计算能量包络
        acc = df[["ax", "ay", "az"]].values
        mag = np.linalg.norm(acc, axis=1)
        env = uniform_filter1d(mag, size=self.smooth_w)

        # 3. 动态阈值 & 掩码
        thresh = np.percentile(env, self.perc)
        active = env > thresh

        # 4. 找不到任何活跃点，返回空列表
        if not active.any():
            return df, []

        # 5. 计算首尾索引（包含）
        start = int(np.argmax(active))
        # 反向查最后一个 True
        rev_idx = int(np.argmax(active[::-1]))
        end = len(active) - 1 - rev_idx

        return df, [(start, end)]


class cut_by_fixed_size(cut_method):
    """
    Cut by fixed size.
    If the tail is not fit to size, just discard it.
    """

    def __init__(self, size=20):
        self.size = size

    def __call__(self, file_path: Path) -> tuple[pd.DataFrame, list]:
        df = pd.read_csv(
            file_path,
            sep=r"\s+",
            header=None,
            names=["ax", "ay", "az", "gx", "gy", "gz"],
        )
        length = len(df["ax"])
        segs, start = [], 0
        for i in range(self.size, length, self.size):
            segs.append((start, i))
            start = i
        return df, segs


class cut_by_segment_file(cut_method):

    def __init__(
        self,
        smooth_w=5,
        perc=75,
        dist_frac=0.3,
    ):
        self.smooth_w = smooth_w
        self.perc = perc
        self.dist_frac = dist_frac

    def __call__(self, file_path: Path) -> tuple[pd.DataFrame, list]:

        def estimate_period(a_env):
            acf = np.correlate(a_env - a_env.mean(), a_env - a_env.mean(), mode="full")
            acf = acf[len(acf) // 2 :]
            peaks, _ = find_peaks(acf, distance=1)
            return peaks[1] if len(peaks) > 1 else len(a_env) // 10

        df = pd.read_csv(
            file_path,
            sep=r"\s+",
            header=None,
            names=["ax", "ay", "az", "gx", "gy", "gz"],
        )
        acc = df[["ax", "ay", "az"]].values
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
            segs.append((start, b))
            start = b
        segs.append((start, len(a_env) - 1))
        return df, segs


class cut_by_change_point(cut_method):
    """
    利用ruptures做变化点检测分段

    Params
    ------
    file_path : str
        IMU 数据文件路径，6列无表头：ax,ay,az,gx,gy,gz
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

    def __call__(self, file_path: Path) -> tuple[pd.DataFrame, list]:

        df = pd.read_csv(
            file_path,
            sep=r"\s+",
            header=None,
            names=["ax", "ay", "az", "gx", "gy", "gz"],
        )
        acc = df[["ax", "ay", "az"]].values

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
            segs.append((start, b))
            start = b
        # 最后一段
        segs.append((start, len(a_env) - 1))
        return df, segs


class cut_by_hmm(cut_method):
    """
    HMM（Hidden Markov Model）：把每个时刻的传感器特征看成观测，隐藏状态分成「静止／运动」两类，利用 HMM 来解码状态序列并提取连续的运动段
    """
    def __init__(
        self,
        n_state: int = 2,
        feature_cols: tuple[str, ...] = ("ax", "ay", "az"),
        cov_type: str = "full",
        random_state: int = 42,
    ):
        self.n_state = n_state
        self.feature_cols = feature_cols
        self.cov_type = cov_type
        self.random_state = random_state

    def __call__(self, file_path: Path) -> tuple[pd.DataFrame, list]:
        # 1) 读取数据
        df = pd.read_csv(
            file_path,
            sep=r"\s+",
            header=None,
            names=["ax", "ay", "az", "gx", "gy", "gz"],
        )
        # 2) 构建观测矩阵：这里只取加速度，也可叠加角速度
        X = df[list(self.feature_cols)].values

        # 3) 训练 HMM
        model = GaussianHMM(
            n_components=self.n_state,
            covariance_type=self.cov_type,
            random_state=self.random_state,
        )
        model.fit(X)

        # 4) 解码状态序列
        states = model.predict(X)  # 每个时刻属于哪个隐藏状态

        # 5) 选取“运动”那一类（假定均值更大）
        means = [model.means_[i].mean() for i in range(self.n_state)]
        move_state = int(np.argmax(means))

        # 6) 把连续的 move_state 段拼成 (start,end)
        segs = []
        active = states == move_state
        idx = np.where(active)[0]
        if idx.size:
            # 找到断点
            splits = np.nonzero(np.diff(idx) > 1)[0]
            starts = [idx[0]] + [idx[i + 1] for i in splits]
            ends = [idx[i] for i in splits] + [idx[-1]]
            segs = [(int(s), int(e)) for s, e in zip(starts, ends)]

        return df, segs


class cut_by_ts_closuring(cut_method):

    """
    Note: should not use this method.
    """
    def __init__(
        self,
        window_size: int = 128,
        step: int = 64,
        n_cluster: int = 2,
        feature_func=None,
    ):
        self.window_size = window_size
        self.step = step
        self.n_cluster = n_cluster
        self.feature_func = feature_func
        # 默认特征：能量均值 & 方差
        if self.feature_func is None:
            self.feature_funcs = [lambda w: w.mean(), lambda w: w.std()]

    def __call__(self, file_path: Path) -> tuple[pd.DataFrame, list]:
        df = pd.read_csv(
            file_path,
            sep=r"\s+",
            header=None,
            names=["ax", "ay", "az", "gx", "gy", "gz"],
        )
        # 1) 提取加速度模长
        mag = np.linalg.norm(df[["ax", "ay", "az"]].values, axis=1)
        N = len(mag)

        # 2) 划小窗并提取特征
        feats = []
        idxs = []
        for i in range(0, N - self.window_size + 1, self.step):
            w = mag[i : i + self.window_size]
            feats.append([f(w) for f in self.feature_funcs])
            idxs.append(i)
        feats = np.array(feats)

        # 3) KMeans 聚成两类
        km = KMeans(n_clusters=self.n_cluster, random_state=0).fit(feats)
        labels = km.labels_

        # 4) 选“运动类”：均值特征更大的那个簇
        cluster_means = feats.mean(axis=1)
        cls_mean = [cluster_means[labels == c].mean() for c in range(self.n_cluster)]
        move_cls = int(np.argmax(cls_mean))

        # 5) 把连续的“运动窗”合并成大段
        active_windows = labels == move_cls
        segs = []
        current = None
        for win_idx, is_active in enumerate(active_windows):
            start_sample = idxs[win_idx]
            end_sample = start_sample + self.window_size - 1
            if is_active:
                if current is None:
                    current = [start_sample, end_sample]
                else:
                    # 如果与上一个窗连在一起，就扩展末尾
                    if start_sample <= current[1] + self.step:
                        current[1] = end_sample
                    else:
                        segs.append(tuple(current))
                        current = [start_sample, end_sample]
            else:
                if current is not None:
                    segs.append(tuple(current))
                    current = None
        if current is not None:
            segs.append(tuple(current))

        return df, segs
