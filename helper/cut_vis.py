import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helper.cut_methods import *
from pathlib import Path
from config import TEST_DATA_DIR
import re

# -------- 配置 --------
sampling_rate = 85.0  # 原始信号采样率
pattern = re.compile(r'\d+')

# 生成一个可调用的切分函数
cut_fn = cut_by_default()

# 读取测试集信息
data = pd.read_csv('39_Test_Dataset/test_info.csv').to_numpy()
test_players_X = []
lens = []

for item in data:
    temp = {}

    unique_id = int(item[0])
    mode = int(item[1]) - 1
    # level = int(item[6]) - 2

    cut = np.array(list(map(int, re.findall(pattern, item[-1]))))
    raw_imu = np.loadtxt(TEST_DATA_DIR / f'{unique_id}.txt')

    _, my_cut = cut_fn(TEST_DATA_DIR / f'{unique_id}.txt')
    lens.append(len(raw_imu))
    if len(raw_imu) < 3000 and len(raw_imu) > 1500:
        continue
    print(f"len: {len(raw_imu)}")
    # raw_imu = (raw_imu - data_mean) / data_std

    # Plot raw_imu data
    plt.figure(figsize=(12, 12))

    # Plot ax, ay, az in the first subplot
    plt.subplot(2, 1, 1)
    for i, label in enumerate(['ax', 'ay', 'az']):
        plt.plot(raw_imu[:, i], label=label)
    # for c in cut:
    #     plt.axvline(x=c, color='red', linestyle='--', linewidth=0.8)
    for (st, ed) in my_cut:
        plt.axvline(x=st, color='purple', linestyle='--', linewidth=0.8)
    plt.axvline(x=my_cut[-1][1], color='purple', linestyle='--', linewidth=0.8)
    plt.title(f'Mode: {mode}')
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.legend()

    # Plot gx, gy, gz in the second subplot
    plt.subplot(2, 1, 2)
    for i, label in enumerate(['gx', 'gy', 'gz']):
        plt.plot(raw_imu[:, i + 3], label=label)
    # for c in cut:
    #     plt.axvline(x=c, color='red', linestyle='--', linewidth=0.8)
    for (st, ed) in my_cut:
        plt.axvline(x=st, color='purple', linestyle='--', linewidth=0.8)
    plt.axvline(x=my_cut[-1][1], color='purple', linestyle='--', linewidth=0.8)
    # plt.title(f'Mode: {mode}')
    plt.xlabel('Time')
    plt.ylabel('Angular Velocity')
    plt.legend()

    plt.tight_layout()

    # plt.xlabel('Time')
    # plt.ylabel('Sensor Values')
    # plt.legend()
    plt.show()
    # show only one
    break

# plt.hist(lens, bins=40)
# plt.show()
