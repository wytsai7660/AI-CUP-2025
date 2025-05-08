import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt

def estimate_period(a_env):
    acf = np.correlate(a_env - a_env.mean(),
                       a_env - a_env.mean(),
                       mode='full')
    acf = acf[len(acf)//2:]
    peaks, _ = find_peaks(acf, distance=1)
    return peaks[1] if len(peaks)>1 else len(a_env)//10

def segment_file(file_path, smooth_w=5, perc=75, dist_frac=0.3):
    df = pd.read_csv(file_path, sep=r'\s+', header=None,
                     names=['ax','ay','az','gx','gy','gz'])
    acc = df[['ax','ay','az']].values
    a_env = uniform_filter1d(np.linalg.norm(acc, axis=1), size=smooth_w)
    T = estimate_period(a_env)
    height = np.percentile(a_env, perc)
    distance = max(int(T * dist_frac), 1)
    peaks, _ = find_peaks(a_env, height=height, distance=distance)
    boundaries = [
        np.argmin(a_env[peaks[i]:peaks[i+1]]) + peaks[i]
        for i in range(len(peaks)-1)
    ]
    segs, start = [], 0
    for b in boundaries:
        segs.append((start, b)); start = b
    segs.append((start, len(a_env)-1))
    return df, segs

# --- 批量绘图 --- #
if __name__ == '__main__':
    data_dir = '39_Training_Dataset/train_data'
    channels = ['ax','ay','az','gx','gy','gz']

    for fname in sorted(os.listdir(data_dir),
                        key=lambda f: int(os.path.splitext(f)[0])):
        if not fname.endswith('.txt'): continue
        path = os.path.join(data_dir, fname)
        df, segs = segment_file(path, smooth_w=5, perc=75, dist_frac=0.3)
        starts = [s for s,_ in segs]
        n_seg = len(segs)

        # 新建画布
        fig, axs = plt.subplots(2, 3, figsize=(14,6), sharex=True)
        axs = axs.flatten()
        time = np.arange(len(df))

        for i, ch in enumerate(channels):
            ax = axs[i]
            ax.plot(time, df[ch], label=ch)
            for j, s in enumerate(starts):
                lbl = 'Segment Start' if j==0 else None
                ax.axvline(s, color='tab:orange',
                        linestyle='--', alpha=0.7, label=lbl)
            ax.set_title(ch)
            ax.grid(True)
            ax.legend(loc='upper right', fontsize='small')

        fig.suptitle(f'{fname} — {n_seg} segments detected')
        plt.tight_layout(rect=[0,0.03,1,0.95])
        plt.show()
        plt.close(fig)
