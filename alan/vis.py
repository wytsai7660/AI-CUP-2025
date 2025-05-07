import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy.signal as signal
import pickle

pattern = re.compile(r'\d+')

# train, val data

data = pd.read_csv('data/train/train_info.csv')
unique_players = data['player_id'].unique()
train_players, test_players = train_test_split(unique_players, test_size=0.2, random_state=42)

data = data.to_numpy()

np.random.shuffle(data)

val_players_X = []
train_players_X = []
val_players_y = []
train_players_y = []

# all_data = []
# for item in data:
#     unique_id = int(item[0])
#     raw_imu = np.loadtxt(f'data/train/train_data/{unique_id}.txt') # (T, 6)
#     all_data.append(raw_imu)

# all_data = np.concatenate(all_data)
# print(all_data.mean(axis=0))
# print(all_data.std(axis=0))
# exit(0)

data_mean = [-628.23490958, -2112.46149982,  -103.7890329,   2669.44510405, 4504.8228582,  -1295.41066289]
data_std = [4682.82936045,  3734.31539544,  2531.04576609, 19961.02033397, 15049.22137978, 21142.38681736]
data_mean = np.array(data_mean)[None, :] # (1, 6)
data_std = np.array(data_std)[None, :] # (1, 6)

# for item in data:
#     temp = {}
    
#     unique_id = int(item[0])
#     player_id = int(item[1])
#     mode = int(item[2]) - 1
#     gender = int(item[3]) - 1
#     hand = int(item[4]) - 1
#     years = int(item[5])
#     level = int(item[6]) - 2
    
#     cut = np.array(list(map(int, re.findall(pattern, item[-1]))))
#     raw_imu = np.loadtxt(f'data/train/train_data/{unique_id}.txt')
#     raw_imu = (raw_imu - data_mean) / data_std
#     # raw_imu = np.sign(raw_imu) * np.log10(np.abs(raw_imu) + 1)
    
#     # Plot raw_imu data
#     plt.figure(figsize=(12, 12))
    
#     # Plot ax, ay, az in the first subplot
#     plt.subplot(2, 1, 1)
#     for i, label in enumerate(['ax', 'ay', 'az']):
#         plt.plot(raw_imu[:, i], label=label)
#     plt.title(f'Mode: {mode}')
#     plt.xlabel('Time')
#     plt.ylabel('Acceleration')
#     plt.legend()
    
#     # Plot gx, gy, gz in the second subplot
#     plt.subplot(2, 1, 2)
#     for i, label in enumerate(['gx', 'gy', 'gz']):
#         plt.plot(raw_imu[:, i + 3], label=label)
#     # plt.title(f'Mode: {mode}')
#     plt.xlabel('Time')
#     plt.ylabel('Angular Velocity')
#     plt.legend()
    
#     plt.tight_layout()

#     # plt.xlabel('Time')
#     # plt.ylabel('Sensor Values')
#     # plt.legend()
#     plt.show()
    
#     plt.figure(figsize=(12, 12))
    
#     # Plot ax, ay, az in the first subplot
#     plt.subplot(2, 1, 1)
#     for i, label in enumerate(['ax', 'ay', 'az']):
#         plt.plot(raw_imu[:, i], label=label)
#     plt.title(f'Years: {years}, Level: {level}, Hand: {hand}, Gender: {gender}, Mode: {mode}')
#     plt.xlabel('Time')
#     plt.ylabel('Acceleration')
#     plt.legend()
    
#     # Plot gx, gy, gz in the second subplot
#     plt.subplot(2, 1, 2)
#     for i, label in enumerate(['gx', 'gy', 'gz']):
#         plt.plot(raw_imu[:, i + 3], label=label)
#     # plt.title(f'Years: {years}, Level: {level}, Hand: {hand}, Gender: {gender}, Mode: {mode}')
#     plt.xlabel('Time')
#     plt.ylabel('Angular Velocity')
#     plt.legend()
    
#     plt.tight_layout()

#     # plt.xlabel('Time')
#     # plt.ylabel('Sensor Values')
#     # plt.legend()
#     plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

def extract_valid_swing_with_fft_percentile(
    data,
    sample_rate=85,
    window_sec=0.5,
    energy_percentile=30, # 新增參數：能量閾值的百分位數 (0-100)
    freq_threshold=0.5,
    freq_band_min=0.5,
    freq_band_max=2
):
    """
    使用 FFT 頻率分析和基於能量百分位數的閾值提取可能的有效揮動區間。

    Args:
        data (np.ndarray): 加速度數據，形狀為 (N, 3)，列順序為 ax, ay, az。
        sample_rate (int): 數據的採樣率 (Hz)。
        window_sec (float): 計算局部能量的滑動窗口大小 (秒)。
        energy_percentile (float): 用於確定能量閾值的百分位數 (0-100)。
        freq_threshold (float): 指定頻段能量佔總能量的閾值 (0-1)。
        freq_band_min (float): 感興趣的頻率下限 (Hz)。
        freq_band_max (float): 感興趣的頻率上限 (Hz)。

    Returns:
        tuple: (trimmed_data, start_idx, end_idx)
            trimmed_data (np.ndarray): 截取後的有效揮動數據。
            start_idx (int): 截取開始的索引。
            end_idx (int): 截取結束的索引。
    """
    # 提取加速度
    ax, ay, az = data[:, 0], data[:, 1], data[:, 2]

    # 計算加速度 Magnitude
    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)

    # 設定滑動窗口
    window_size = int(window_sec * sample_rate)
    if window_size == 0: # 避免窗口大小為0
        window_size = 1

    # 計算局部能量 (平滑)
    # 使用 'valid' 模式避免邊緣效應，或者確保窗口足夠小
    # 使用 'same' 模式保持與原始數據長度一致
    energy = np.convolve(acc_mag**2, np.ones(window_size)/window_size, mode='same')

    # 計算基於百分位數的能量閾值
    if len(energy) > 0:
        dynamic_energy_threshold = np.percentile(energy, energy_percentile)
        print(f"Calculated energy threshold ({energy_percentile}th percentile): {dynamic_energy_threshold:.4f}")
    else:
        print("Warning: Energy array is empty. Cannot calculate percentile threshold.")
        dynamic_energy_threshold = 0 # 設置為0， active 將會全 False

    # FFT 頻率分析
    N = len(acc_mag)
    if N == 0:
         print("Warning: Data length is 0. Cannot perform FFT.")
         return data, 0, 0

    # 去掉直流分量 (平均值)
    acc_mag_detrended = acc_mag - np.mean(acc_mag)

    yf = rfft(acc_mag_detrended)
    xf = rfftfreq(N, 1 / sample_rate)

    # 計算指定頻段的頻率能量
    # 確保頻率範圍有效
    freq_band_min = max(0, freq_band_min) # 頻率不能小於0
    freq_band = (xf >= freq_band_min) & (xf <= freq_band_max)

    # 確保 freq_band 不為空，避免 sum of empty slice warning
    if not np.any(freq_band):
        freq_energy = 0
        print(f"Warning: No frequencies found in band {freq_band_min}-{freq_band_max}Hz.")
    else:
        freq_energy = np.sum(np.abs(yf[freq_band])**2)

    freq_energy_total = np.sum(np.abs(yf)**2) # 總能量

    freq_energy_normalized = freq_energy / freq_energy_total if freq_energy_total > 0 else 0

    print(f"Normalized swing band ({freq_band_min}-{freq_band_max}Hz) energy: {freq_energy_normalized:.4f}")

    # 雙重條件判斷：頻率集中 && 能量夠高
    # 首先判斷頻率是否符合要求
    if freq_energy_normalized > freq_threshold:
        # 如果頻率符合，則基於能量閾值找出活躍區間
        active = energy > dynamic_energy_threshold

        if np.any(active):
            # 找到第一個 True 和最後一個 True 的索引
            start_idx = np.argmax(active)
            # np.argmax(active[::-1]) 找到反轉後第一個 True 的索引
            # len(active) - 1 - index 得到原始數組中對應的索引
            # 因為我們要的是區間結束後一個位置，所以是 len(active) - index
            end_idx = len(active) - np.argmax(active[::-1])
        else:
            # 如果能量閾值太高，沒有任何點超過閾值
            print("Warning: Frequency condition met, but no points exceeded the energy percentile threshold.")
            start_idx, end_idx = 0, len(data) # 或者可以考慮返回 0, 0 或 None

    else:
        # 如果頻率條件不符合，認為沒有有效揮動
        print("Warning: Frequency energy too low, possibly no valid swing detected.")
        start_idx, end_idx = 0, len(data) # 或者可以考慮返回 0, 0 或 None

    trimmed_data = data[start_idx:end_idx]

    # --- 畫圖展示 ---
    fig, axs = plt.subplots(3, 1, figsize=(15,10))

    # 加速度 magnitude
    axs[0].plot(acc_mag, label='Acc Magnitude')
    axs[0].axvline(start_idx, color='g', linestyle='--', label='Start')
    axs[0].axvline(end_idx, color='r', linestyle='--', label='End')
    axs[0].set_title('Acceleration Magnitude')
    axs[0].legend()

    # Energy
    axs[1].plot(energy, label='Sliding Energy', color='orange')
    # 繪製動態計算出的能量閾值
    axs[1].axhline(dynamic_energy_threshold, color='gray', linestyle='--', label=f'Energy Threshold ({energy_percentile}th percentile)')
    axs[1].axvline(start_idx, color='g', linestyle='--')
    axs[1].axvline(end_idx, color='r', linestyle='--')
    axs[1].set_title('Energy (Sliding Window)')
    axs[1].legend()

    # FFT
    axs[2].plot(xf, np.abs(yf)**2, label='FFT Energy Spectrum', color='purple')
    axs[2].axvspan(freq_band_min, freq_band_max, color='yellow', alpha=0.3, label=f'Freq Band ({freq_band_min}-{freq_band_max}Hz)')
    # 設置合理的X軸範圍，例如顯示到採樣率的一半或更少
    axs[2].set_xlim(0, sample_rate / 2) # 顯示 Nyquist 頻率範圍
    axs[2].set_xlabel('Frequency (Hz)')
    axs[2].set_ylabel('Energy')
    axs[2].set_title('FFT Frequency Analysis')
    axs[2].legend()

    plt.tight_layout()
    plt.show()

    return trimmed_data, start_idx, end_idx

from scipy.signal import detrend

def extract_valid_swing_with_local_fft(
    data,
    sample_rate=85,
    energy_window_sec=0.5,
    energy_percentile=30, # 能量閾值的百分位數
    fft_window_sec=3.0,  # 新增：局部 FFT 分析的窗口大小 (秒)
    fft_step_sec=0.1,    # 新增：局部 FFT 分析的窗口步長 (秒)
    local_freq_threshold=0.4, # 新增：局部頻率能量佔比閾值
    freq_band_min=0.5,
    freq_band_max=2,
    global_freq_threshold=0.5 # 保留：整體頻率能量佔比閾值
):
    """
    使用基於能量百分位數的閾值和局部 FFT 頻率分析，
    提取可能的有效揮動區間，並嘗試排除頭尾的靜止或亂動。

    Args:
        data (np.ndarray): 加速度數據，形狀為 (N, 3)，列順序為 ax, ay, az。
        sample_rate (int): 數據的採樣率 (Hz)。
        energy_window_sec (float): 計算局部能量的滑動窗口大小 (秒)。
        energy_percentile (float): 用於確定能量閾值的百分位數 (0-100)。
        fft_window_sec (float): 局部 FFT 分析的窗口大小 (秒)。
        fft_step_sec (float): 局部 FFT 分析的窗口步長 (秒)。
        local_freq_threshold (float): 局部窗口內，指定頻段能量佔總能量的閾值 (0-1)。
        freq_band_min (float): 感興趣的頻率下限 (Hz)。
        freq_band_max (float): 感興趣的頻率上限 (Hz)。
        global_freq_threshold (float): 整個數據，指定頻段能量佔總能量的閾值 (0-1)。

    Returns:
        tuple: (trimmed_data, start_idx, end_idx)
            trimmed_data (np.ndarray): 截取後的有效揮動數據。
            start_idx (int): 截取開始的索引。
            end_idx (int): 截取結束的索引。
    """
    N_total = len(data)
    if N_total == 0:
         print("Warning: Data length is 0.")
         return data, 0, 0

    # 提取加速度
    ax, ay, az = data[:, 0], data[:, 1], data[:, 2]

    # 計算加速度 Magnitude
    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)

    # --- 步驟 1: 全局頻率檢查 (快速判斷整體數據是否包含揮動特徵) ---
    acc_mag_detrended_global = acc_mag - np.mean(acc_mag)
    yf_global = rfft(acc_mag_detrended_global)
    xf_global = rfftfreq(N_total, 1 / sample_rate)

    freq_band_global = (xf_global >= freq_band_min) & (xf_global <= freq_band_max)
    if not np.any(freq_band_global):
         global_freq_energy = 0
         print(f"Warning: No frequencies found in global band {freq_band_min}-{freq_band_max}Hz.")
    else:
        global_freq_energy = np.sum(np.abs(yf_global[freq_band_global])**2)

    global_freq_energy_total = np.sum(np.abs(yf_global)**2)
    global_freq_energy_normalized = global_freq_energy / global_freq_energy_total if global_freq_energy_total > 0 else 0

    print(f"Global swing band ({freq_band_min}-{freq_band_max}Hz) normalized energy: {global_freq_energy_normalized:.4f}")

    # 如果整體頻率能量不足，則認為沒有有效揮動，返回原始數據或空數據
    if global_freq_energy_normalized < global_freq_threshold:
        print("Warning: Global frequency energy too low, likely no valid swing detected.")
        # 可以選擇返回原始數據範圍，或者返回 0, 0 表示未找到
        return data, 0, N_total # 返回原始數據範圍

    print("Global frequency threshold met. Proceeding with local analysis...")

    # --- 步驟 2: 計算局部能量 ---
    energy_window_size = int(energy_window_sec * sample_rate)
    if energy_window_size == 0: energy_window_size = 1
    energy = np.convolve(acc_mag**2, np.ones(energy_window_size)/energy_window_size, mode='same')

    # 計算基於百分位數的能量閾值
    if len(energy) > 0:
        dynamic_energy_threshold = np.percentile(energy, energy_percentile)
        print(f"Calculated energy threshold ({energy_percentile}th percentile): {dynamic_energy_threshold:.4f}")
    else:
        dynamic_energy_threshold = 0
        print("Warning: Energy array is empty. Cannot calculate percentile threshold.")

    # --- 步驟 3: 局部 FFT 頻率分析 ---
    fft_window_size = int(fft_window_sec * sample_rate)
    fft_step_size = int(fft_step_sec * sample_rate)

    if fft_window_size == 0: fft_window_size = 1
    if fft_step_size == 0: fft_step_size = 1
    if fft_window_size > N_total:
         print(f"Warning: FFT window size ({fft_window_size}) larger than data length ({N_total}). Using data length as window size.")
         fft_window_size = N_total
         fft_step_size = N_total # Only one window

    # 存儲每個窗口的頻率分析結果
    local_freq_active_windows = []
    # 存儲每個窗口對應的原始數據中心索引 (用於後續對齊)
    window_center_indices = []

    # 滑動窗口進行 FFT
    for i in range(0, N_total - fft_window_size + 1, fft_step_size):
        window_data = acc_mag[i : i + fft_window_size]

        # 對窗口數據去趨勢 (重要，消除靜止時的 DC 偏移或慢速漂移)
        window_data_detrended = detrend(window_data) # 使用 scipy.signal.detrend

        if len(window_data_detrended) == 0: continue

        # 局部 FFT
        yf_local = rfft(window_data_detrended)
        xf_local = rfftfreq(len(window_data_detrended), 1 / sample_rate)

        # 計算局部窗口內指定頻段的能量
        freq_band_local = (xf_local >= freq_band_min) & (xf_local <= freq_band_max)
        if not np.any(freq_band_local):
            local_freq_energy = 0
        else:
            local_freq_energy = np.sum(np.abs(yf_local[freq_band_local])**2)

        local_freq_energy_total = np.sum(np.abs(yf_local)**2)

        local_freq_energy_normalized = local_freq_energy / local_freq_energy_total if local_freq_energy_total > 0 else 0

        # 判斷局部窗口是否符合頻率特徵
        is_local_swing_freq = local_freq_energy_normalized > local_freq_threshold
        local_freq_active_windows.append(is_local_swing_freq)

        # 計算窗口的中心索引 (用於後續對齊到原始數據索引)
        window_center_index = i + fft_window_size // 2
        window_center_indices.append(window_center_index)

    # 如果沒有窗口通過局部頻率檢查，則認為沒有有效揮動
    if not local_freq_active_windows:
         print("Warning: No local windows passed the frequency threshold.")
         return data, 0, N_total # 或返回 0, 0

    # 將局部頻率激活結果上採樣到原始數據長度，以便與能量陣列對齊
    # 這裡使用一個簡單的插值方法，將窗口中心點的激活狀態擴展到窗口範圍
    local_freq_active_upsampled = np.zeros(N_total, dtype=bool)
    for i in range(len(window_center_indices)):
        center_idx = window_center_indices[i]
        is_active = local_freq_active_windows[i]
        # 將窗口的激活狀態影響範圍擴展到窗口的開始到結束索引
        start_idx_window = max(0, center_idx - fft_window_size // 2)
        end_idx_window = min(N_total, center_idx + fft_window_size // 2 + 1) # +1 for slicing end not included
        local_freq_active_upsampled[start_idx_window : end_idx_window] = is_active


    # --- 步驟 4: 結合能量閾值和局部頻率激活 ---
    # 只有當能量夠高 *並且* 局部頻率特徵符合時，才視為活躍點
    # 注意：能量陣列和 local_freq_active_upsampled 應該長度一致
    active = (energy > dynamic_energy_threshold) & local_freq_active_upsampled[:len(energy)] # 確保長度一致


    # --- 步驟 5: 找出活躍區間的起始和結束索引 ---
    if np.any(active):
        start_idx = np.argmax(active) # 第一個 True 的索引
        # 找到反轉後第一個 True 的索引，計算出原始數組中對應的最後一個 True 的後一個位置
        end_idx = len(active) - np.argmax(active[::-1])
    else:
        # 如果沒有任何點同時滿足能量和頻率條件
        print("Warning: No points met both energy and local frequency criteria.")
        start_idx, end_idx = 0, N_total # 或者可以考慮返回 0, 0

    trimmed_data = data[start_idx:end_idx]

    # --- 畫圖展示 ---
    fig, axs = plt.subplots(4, 1, figsize=(15,12))

    # 加速度 magnitude
    axs[0].plot(acc_mag, label='Acc Magnitude')
    axs[0].axvline(start_idx, color='g', linestyle='--', label='Start')
    axs[0].axvline(end_idx, color='r', linestyle='--', label='End')
    axs[0].set_title('Acceleration Magnitude')
    axs[0].legend()
    axs[0].grid(True)

    # Energy and Threshold
    axs[1].plot(energy, label='Sliding Energy', color='orange')
    axs[1].axhline(dynamic_energy_threshold, color='gray', linestyle='--', label=f'Energy Threshold ({energy_percentile}th percentile)')
    axs[1].axvline(start_idx, color='g', linestyle='--')
    axs[1].axvline(end_idx, color='r', linestyle='--')
    axs[1].set_title('Energy (Sliding Window)')
    axs[1].legend()
    axs[1].grid(True)

    # Local Frequency Activation
    # 繪製上採樣後的局部頻率激活狀態
    axs[2].plot(local_freq_active_upsampled, label='Local Freq Activation (Upsampled)', color='purple', drawstyle='steps-post')
    axs[2].axvline(start_idx, color='g', linestyle='--')
    axs[2].axvline(end_idx, color='r', linestyle='--')
    axs[2].set_title(f'Local Frequency Band ({freq_band_min}-{freq_band_max}Hz) Activation (Threshold > {local_freq_threshold:.2f})')
    axs[2].set_yticks([0, 1])
    axs[2].set_yticklabels(['Inactive', 'Active'])
    axs[2].legend()
    axs[2].grid(True)


    # Combined Activation (Final 'active' array)
    axs[3].plot(active, label='Combined Activation (Energy & Freq)', color='brown', drawstyle='steps-post')
    axs[3].axvline(start_idx, color='g', linestyle='--', label='Start')
    axs[3].axvline(end_idx, color='r', linestyle='--', label='End')
    axs[3].set_title('Combined Activation (Energy AND Local Frequency)')
    axs[3].set_yticks([0, 1])
    axs[3].set_yticklabels(['Inactive', 'Active'])
    axs[3].legend()
    axs[3].grid(True)


    plt.tight_layout()
    plt.show()

    # 可選：繪製全局 FFT
    plt.figure(figsize=(10, 4))
    plt.plot(xf_global, np.abs(yf_global)**2, label='Global FFT Energy Spectrum', color='blue')
    plt.axvspan(freq_band_min, freq_band_max, color='yellow', alpha=0.3, label=f'Global Freq Band ({freq_band_min}-{freq_band_max}Hz)')
    plt.xlim(0, sample_rate / 2) # 顯示 Nyquist 頻率範圍
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Energy')
    plt.title('Global FFT Frequency Analysis')
    plt.legend()
    plt.grid(True)
    plt.show()


    return trimmed_data, start_idx, end_idx


# --- 使用範例 ---
# 假設 raw_data 是你的 (T,6) numpy array
# trimmed_data, start_idx, end_idx = extract_valid_swing_with_fft(raw_data)


# test data

data = pd.read_csv('data/test/test_info.csv')
data = data.to_numpy()

test_players_X = []
lens = []

for item in data:
    temp = {}
    
    unique_id = int(item[0])
    mode = int(item[1]) - 1
    # level = int(item[6]) - 2
    
    cut = np.array(list(map(int, re.findall(pattern, item[-1]))))
    raw_imu = np.loadtxt(f'data/test/test_data/{unique_id}.txt')
    lens.append(len(raw_imu))
    if len(raw_imu) < 3000 and len(raw_imu) > 1500:
        continue
    print(f"len: {len(raw_imu)}")
    # raw_imu = (raw_imu - data_mean) / data_std
    
    result = extract_valid_swing_with_local_fft(raw_imu)
    
    # Plot raw_imu data
    plt.figure(figsize=(12, 12))
    
    # Plot ax, ay, az in the first subplot
    plt.subplot(2, 1, 1)
    for i, label in enumerate(['ax', 'ay', 'az']):
        plt.plot(raw_imu[:, i], label=label)
    for c in cut:
        plt.axvline(x=c, color='red', linestyle='--', linewidth=0.8)

    plt.title(f'Mode: {mode}')
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.legend()
    
    # Plot gx, gy, gz in the second subplot
    plt.subplot(2, 1, 2)
    for i, label in enumerate(['gx', 'gy', 'gz']):
        plt.plot(raw_imu[:, i + 3], label=label)
    for c in cut:
        plt.axvline(x=c, color='red', linestyle='--', linewidth=0.8)

    # plt.title(f'Mode: {mode}')
    plt.xlabel('Time')
    plt.ylabel('Angular Velocity')
    plt.legend()
    
    plt.tight_layout()

    # plt.xlabel('Time')
    # plt.ylabel('Sensor Values')
    # plt.legend()
    plt.show()

plt.hist(lens, bins=40)
plt.show()