from pathlib import Path
import numpy as np
import pandas as pd
import math
import csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import os
from config import TRAIN_DATA_DIR, TRAIN_INFO, BASE_PATH
import torch
from helper.segment import Trim, Yungan, ChangePoint, IntergratedSplit
import warnings
from scipy.signal import detrend


def FFT(xreal, ximag):
    # Combine real and imaginary parts into a complex array
    x = np.array(xreal) + 1j * np.array(ximag)
    # Find the largest power of 2 less than or equal to len(x)
    n = 2
    while n * 2 <= len(x):
        n *= 2
    x = x[:n]
    # Perform FFT using numpy
    X = np.fft.fft(x)
    # Return real and imaginary parts
    return n, X.real.tolist(), X.imag.tolist()
    
def FFT_data(input_data, swinging_times):
    input_data = np.asarray(input_data)
    txtlength = swinging_times[-1] - swinging_times[0]
    a_mean = np.zeros(txtlength)
    g_mean = np.zeros(txtlength)

    for i in range(len(swinging_times) - 1):
        seg = input_data[swinging_times[i]:swinging_times[i + 1]]
        if seg.shape[0] == 0:
            continue
        a = np.linalg.norm(seg[:, 0:3], axis=1)
        g = np.linalg.norm(seg[:, 3:6], axis=1)
        a_mean[i] = np.mean(a)
        g_mean[i] = np.mean(g)

    return a_mean.tolist(), g_mean.tolist()

def feature(input_data, swinging_now, swinging_times, n_fft, a_fft, g_fft, a_fft_imag, g_fft_imag, writer):
    allsum = []
    mean = []
    var = []
    rms = []
    XYZmean_a = 0
    a = []
    g = []
    a_s1 = 0
    a_s2 = 0
    g_s1 = 0
    g_s2 = 0
    a_k1 = 0
    a_k2 = 0
    g_k1 = 0
    g_k2 = 0
    
    for i in range(len(input_data)):
        if i==0:
            allsum = input_data[i]
            a.append(math.sqrt(math.pow((input_data[i][0] + input_data[i][1] + input_data[i][2]), 2)))
            g.append(math.sqrt(math.pow((input_data[i][3] + input_data[i][4] + input_data[i][5]), 2)))
            continue
        
        a.append(math.sqrt(math.pow((input_data[i][0] + input_data[i][1] + input_data[i][2]), 2)))
        g.append(math.sqrt(math.pow((input_data[i][3] + input_data[i][4] + input_data[i][5]), 2)))
       
        allsum = [allsum[feature_index] + input_data[i][feature_index] for feature_index in range(len(input_data[i]))]
        
    mean = [allsum[feature_index] / len(input_data) for feature_index in range(len(input_data[i]))]
    
    for i in range(len(input_data)):
        if i==0:
            var = input_data[i]
            rms = input_data[i]
            continue

        var = [var[feature_index] + math.pow((input_data[i][feature_index] - mean[feature_index]), 2) for feature_index in range(len(input_data[i]))]
        rms = [rms[feature_index] + math.pow(input_data[i][feature_index], 2) for feature_index in range(len(input_data[i]))]
        
    var = [math.sqrt((var[feature_index] / len(input_data))) for feature_index in range(len(input_data[i]))]
    rms = [math.sqrt((rms[feature_index] / len(input_data))) for feature_index in range(len(input_data[i]))]
    
    a_max = [max(a)]
    a_min = [min(a)]
    a_mean = [sum(a) / len(a)]
    g_max = [max(g)]
    g_min = [min(g)]
    g_mean = [sum(g) / len(g)]
    
    a_var = math.sqrt(math.pow((var[0] + var[1] + var[2]), 2))
    
    for i in range(len(input_data)):
        a_s1 = a_s1 + math.pow((a[i] - a_mean[0]), 4)
        a_s2 = a_s2 + math.pow((a[i] - a_mean[0]), 2)
        g_s1 = g_s1 + math.pow((g[i] - g_mean[0]), 4)
        g_s2 = g_s2 + math.pow((g[i] - g_mean[0]), 2)
        a_k1 = a_k1 + math.pow((a[i] - a_mean[0]), 3)
        g_k1 = g_k1 + math.pow((g[i] - g_mean[0]), 3)
    
    a_s1 = a_s1 / len(input_data)
    a_s2 = a_s2 / len(input_data)
    g_s1 = g_s1 / len(input_data)
    g_s2 = g_s2 / len(input_data)
    a_k2 = math.pow(a_s2, 1.5)
    g_k2 = math.pow(g_s2, 1.5)
    a_s2 = a_s2 * a_s2
    g_s2 = g_s2 * g_s2
    
    a_kurtosis = [a_s1 / a_s2]
    g_kurtosis = [g_s1 / g_s2]
    a_skewness = [a_k1 / a_k2]
    g_skewness = [g_k1 / g_k2]
    
    a_fft_mean = 0
    g_fft_mean = 0
    cut = int(n_fft / swinging_times)
    a_psd = []
    g_psd = []
    entropy_a = []
    entropy_g = []
    e1 = []
    e3 = []
    e2 = 0
    e4 = 0
    
    for i in range(cut * swinging_now, cut * (swinging_now + 1)):
        a_fft_mean += a_fft[i]
        g_fft_mean += g_fft[i]
        a_psd.append(math.pow(a_fft[i], 2) + math.pow(a_fft_imag[i], 2))
        g_psd.append(math.pow(g_fft[i], 2) + math.pow(g_fft_imag[i], 2))
        e1.append(math.pow(a_psd[-1], 0.5))
        e3.append(math.pow(g_psd[-1], 0.5))
        
    a_fft_mean = a_fft_mean / cut
    g_fft_mean = g_fft_mean / cut
    
    a_psd_mean = sum(a_psd) / len(a_psd)
    g_psd_mean = sum(g_psd) / len(g_psd)
    
    for i in range(cut):
        e2 += math.pow(a_psd[i], 0.5)
        e4 += math.pow(g_psd[i], 0.5)
    
    for i in range(cut):
        entropy_a.append((e1[i] / e2) * math.log(e1[i] / e2))
        entropy_g.append((e3[i] / e4) * math.log(e3[i] / e4))
    
    a_entropy_mean = sum(entropy_a) / len(entropy_a)
    g_entropy_mean = sum(entropy_g) / len(entropy_g)       
        
    
    output = mean + var + rms + a_max + a_mean + a_min + g_max + g_mean + g_min + [a_fft_mean] + [g_fft_mean] + [a_psd_mean] + [g_psd_mean] + a_kurtosis + g_kurtosis + a_skewness + g_skewness + [a_entropy_mean] + [g_entropy_mean]
    writer.writerow(output)

def data_generate(datapath: Path, tar_dir: Path):
    datapath = '../39_Training_Dataset/train_data'
    tar_dir = '../39_Training_Dataset/tabular_data_train'
    headerList = ['ax_mean', 'ay_mean', 'az_mean', 'gx_mean', 'gy_mean', 'gz_mean', 'ax_var', 'ay_var', 'az_var', 'gx_var', 'gy_var', 'gz_var', 'ax_rms', 'ay_rms', 'az_rms', 'gx_rms', 'gy_rms', 'gz_rms', 'a_max', 'a_mean', 'a_min', 'g_max', 'g_mean', 'g_min', 'a_fft', 'g_fft', 'a_psd', 'g_psd', 'a_kurt', 'g_kurt', 'a_skewn', 'g_skewn', 'a_entropy', 'g_entropy']                

    os.makedirs(tar_dir, exist_ok=True)
    pathlist_txt = Path(datapath).glob('**/*.txt')
        
    for file in pathlist_txt:
        f = open(file)

        All_data = []

        count = 0
        for line in f.readlines():
            if line == '\n' or count == 0:
                count += 1
                continue
            num = line.split(' ')
            if len(num) > 5:
                tmp_list = []
                for i in range(6):
                    tmp_list.append(int(num[i]))
                All_data.append(tmp_list)
        
        f.close()

        swing_index = np.linspace(0, len(All_data), 28, dtype = int)
        # filename.append(int(Path(file).stem))
        # all_swing.append([swing_index])

        headerList = ['ax_mean', 'ay_mean', 'az_mean', 'gx_mean', 'gy_mean', 'gz_mean', 'ax_var', 'ay_var', 'az_var', 'gx_var', 'gy_var', 'gz_var', 'ax_rms', 'ay_rms', 'az_rms', 'gx_rms', 'gy_rms', 'gz_rms', 'a_max', 'a_mean', 'a_min', 'g_max', 'g_mean', 'g_min', 'a_fft', 'g_fft', 'a_psd', 'g_psd', 'a_kurt', 'g_kurt', 'a_skewn', 'g_skewn', 'a_entropy', 'g_entropy']                
        

        with open('{dir}/{fname}.csv'.format(dir = tar_dir, fname = Path(file).stem), 'w', newline = '') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headerList)
            try:
                a_fft, g_fft = FFT_data(All_data, swing_index)
                a_fft_imag = [0] * len(a_fft)
                g_fft_imag = [0] * len(g_fft)
                n_fft, a_fft, a_fft_imag = FFT(a_fft, a_fft_imag)
                n_fft, g_fft, g_fft_imag = FFT(g_fft, g_fft_imag)
                for i in range(len(swing_index)):
                    if i==0:
                        continue
                    feature(All_data[swing_index[i-1]: swing_index[i]], i - 1, len(swing_index) - 1, n_fft, a_fft, g_fft, a_fft_imag, g_fft_imag, writer)
            except:
                print(Path(file).stem)
                continue

def feature_gen(data, writer):
    a_mag = np.linalg.norm(data[:, 0:3], axis=1)
    g_mag = np.linalg.norm(data[:, 3:6], axis=1)
    # data = np.column_stack((data, a_mag, g_mag))
    data = np.column_stack((a_mag, g_mag))
    means = np.mean(data, axis=0)
    vars = np.var(data, axis=0)
    rms = np.sqrt(np.mean(data**2, axis=0))
    maxs = np.max(data, axis=0)
    mins = np.min(data, axis=0)
    output = means.tolist() + vars.tolist() + rms.tolist() + maxs.tolist() + mins.tolist()
    # output = maxs.tolist() + vars.tolist()
    writer.writerow(output)
    

def generate_csv(info: pd.DataFrame, tar_dir: Path):
    os.makedirs(tar_dir, exist_ok=True)
    unique_ids = info['unique_id'].values
    
    fpaths = [
            TRAIN_DATA_DIR / f"{unique_id}.txt" for unique_id in unique_ids
        ]
    
    
    for file, unique_id in zip(fpaths, unique_ids):
        data = np.loadtxt(file, dtype = np.float32)
        data = detrend(data)
        data_torch = torch.tensor(data, dtype=torch.float32)
        trim_method = Trim()
        # segment_method = ChangePoint()
        data_torch = trim_method(data_torch)[0]
        # swing_index = segment_method(data_torch)
        data = data_torch.numpy()
        # print(len(swing_index))
        
        swing_index = np.linspace(0, len(data), 28, dtype = int)
        
        with open(tar_dir / f"{unique_id}.csv", 'w', newline = '') as csvfile:
            writer = csv.writer(csvfile)
            # headerList = ['ax_mean', 'ay_mean', 'az_mean', 'gx_mean', 'gy_mean', 'gz_mean', 'ax_var', 'ay_var', 'az_var', 'gx_var', 'gy_var', 'gz_var', 'ax_rms', 'ay_rms', 'az_rms', 'gx_rms', 'gy_rms', 'gz_rms', 'a_max', 'a_mean', 'a_min', 'g_max', 'g_mean', 'g_min']
            # headerList = ['ax_max', 'ay_max', 'az_max', 'gx_max', 'gy_max', 'gz_max', 'a_max', 'g_max', 'ax_var', 'ay_var', 'az_var', 'gx_var', 'gy_var', 'gz_var', 'a_var', 'g_var']
            headerList = ['a_mean', 'g_mean', 'a_var', 'g_var', 'a_rms', 'g_rms', 'a_max', 'g_max', 'a_min', 'g_min']

            # headerList = ['ax_mean', 'ay_mean', 'az_mean', 'gx_mean', 'gy_mean', 'gz_mean', 'ax_var', 'ay_var', 'az_var', 'gx_var', 'gy_var', 'gz_var', 'ax_rms', 'ay_rms', 'az_rms', 'gx_rms', 'gy_rms', 'gz_rms', 'a_max', 'a_mean', 'a_min', 'g_max', 'g_mean', 'g_min', 'a_fft', 'g_fft', 'a_psd', 'g_psd', 'a_kurt', 'g_kurt', 'a_skewn', 'g_skewn', 'a_entropy', 'g_entropy']                
            writer.writerow(headerList)
            
            try:
                for s, e in zip(swing_index, swing_index[1:]):
                    feature_gen(data[s:e], writer)
            except Exception as e:
                print(f"Error processing unique_id {unique_id}: {e}")
                continue

            # 這裡可以加入生成 CSV 的邏輯
            # 例如：讀取檔案、計算特徵、寫入 CSV 等等
    

def main():
    # 若尚未產生特徵，請先執行 data_generate() 生成特徵 CSV 檔案
    info = pd.read_csv(TRAIN_INFO)
    np.random.seed(42)
    torch.manual_seed(42)
    generate_csv(info, Path('39_Training_Dataset/tabular2_data_train'))
    # data_generate()
    # exit(0)
    
    # 讀取訓練資訊，根據 player_id 將資料分成 80% 訓練、20% 測試
    # unique_players = info['player_id'].unique()
    # train_players, test_players = train_test_split(unique_players, test_size=0.2, random_state=42)
    unique_player = info.drop_duplicates(subset=["player_id"])

    train_players, test_players = train_test_split(
        unique_player["player_id"].to_numpy(),
        test_size=0.2,
        random_state=42,
        stratify=unique_player['level'].to_numpy(),
    )
    
    # 讀取特徵 CSV 檔（位於 "./tabular_data_train"）
    datapath = Path('39_Training_Dataset/tabular_data_train')
    target_mask = ['gender', 'hold racket handed', 'play years', 'level']
    
    # 根據 test_players 分組資料
    x_train = pd.DataFrame()
    y_train = pd.DataFrame(columns=target_mask)
    x_test = pd.DataFrame()
    y_test = pd.DataFrame(columns=target_mask)
    y_test_split_point = [0]
    
    for file in datapath.iterdir():
        unique_id = int(Path(file).stem)
        row = info[info['unique_id'] == unique_id]
        if row.empty:
            continue
        player_id = row['player_id'].iloc[0]
        data = pd.read_csv(file)
        
        target = row[target_mask]
        try:
            target_repeated = pd.concat([target] * len(data))
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue
        if player_id in train_players:
            x_train = pd.concat([x_train, data], ignore_index=True)
            y_train = pd.concat([y_train, target_repeated], ignore_index=True)
        elif player_id in test_players:
            x_test = pd.concat([x_test, data], ignore_index=True)
            y_test = pd.concat([y_test, target_repeated], ignore_index=True)
            y_test_split_point.append(len(data))
    
    y_test_split_point = np.cumsum(y_test_split_point)
    print(f"Train data shape: {x_train.shape}, Test data shape: {x_test.shape}")
    print(f"Train target shape: {y_train.shape}, Test target shape: {y_test.shape}")
    
    # 標準化特徵
    scaler = MinMaxScaler(feature_range=(0, 1))
    le = LabelEncoder()
    X_train_scaled = scaler.fit_transform(x_train)
    X_test_scaled = scaler.transform(x_test)
    
    group_size = 27

    def model_binary(X_train, y_train, X_test, y_test):
        clf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=8)
        clf.fit(X_train, y_train)
        
        predicted = clf.predict_proba(X_test)
        predicted = predicted[:, 0]
        # 取出正類（index 0）的概率
        # predicted = [predicted[i][0] for i in range(len(predicted))]
        
        y_pred = []
        y_test_agg = []
        for s, e in zip(y_test_split_point, y_test_split_point[1:]):
            predicted[s:e] = 1 - predicted[s:e]
            y_pred.append(np.min(predicted[s:e]).item())
            y_test_agg.append(y_test[s])
            # y_pred.append(np.mean(predicted[s:e]).item())
        # num_groups = len(predicted) // group_size 
        # if sum(predicted[:group_size]) / group_size > 0.5:
        #     y_pred = [max(predicted[i*group_size: (i+1)*group_size]) for i in range(num_groups)]
        # else:
        #     y_pred = [min(predicted[i*group_size: (i+1)*group_size]) for i in range(num_groups)]
        
        # y_pred  = [1 - x for x in y_pred]
        # y_test_agg = [y_test[i*group_size] for i in range(num_groups)]
        auc_score = roc_auc_score(y_test_agg, y_pred, average='micro')
        f1 = f1_score(y_test_agg, [round(x) for x in y_pred], average='macro')
        acc = accuracy_score(y_test_agg, [round(x) for x in y_pred])
        print('Binary F1:', f1)
        print('Binary ACC:', acc)
        print('Binary AUC:', auc_score)
        # exit(0)

    # 定義多類別分類評分函數 (例如 play years、level)
    def model_multiary(X_train, y_train, X_test, y_test):
        clf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=8)
        clf.fit(X_train, y_train)
        predicted = clf.predict_proba(X_test)
        
        
        num_classes = len(np.unique(y_train))
        num_groups = len(predicted) // group_size
        y_pred = []
        y_test_agg = []
        for s, e in zip(y_test_split_point, y_test_split_point[1:]):
            group_pred = predicted[s:e]
            class_sums = np.sum(group_pred, axis=0)
            chosen_class = np.argmax(class_sums)
            candidate_probs = group_pred[:, chosen_class]
            best_instance = np.argmax(candidate_probs)
            y_pred.append(group_pred[best_instance])
            y_test_agg.append(y_test[s])
        # for i in range(num_groups):
        #     group_pred = predicted[i*group_size: (i+1)*group_size]
        #     num_classes = len(np.unique(y_train))
        #     # 對每個類別計算該組內的總機率
        #     class_sums = [sum([group_pred[k][j] for k in range(group_size)]) for j in range(num_classes)]
        #     chosen_class = np.argmax(class_sums)
        #     candidate_probs = [group_pred[k][chosen_class] for k in range(group_size)]
        #     best_instance = np.argmax(candidate_probs)
        #     y_pred.append(group_pred[best_instance])
        
        # y_test_agg = [y_test[i*group_size] for i in range(num_groups)]
        auc_score = roc_auc_score(y_test_agg, y_pred, average='micro', multi_class='ovr')
        f1 = f1_score(y_test_agg, np.argmax(y_pred, axis=1), average='macro')
        acc = accuracy_score(y_test_agg, np.argmax(y_pred, axis=1))
        print('Multiary F1:', f1)
        print('Multiary ACC:', acc)
        print('Multiary AUC:', auc_score)

    # 評分：針對各目標進行模型訓練與評分
    y_train_le_gender = le.fit_transform(y_train['gender'])
    y_test_le_gender = le.transform(y_test['gender'])
    model_binary(X_train_scaled, y_train_le_gender, X_test_scaled, y_test_le_gender)
    print('----------------------------------')
    
    y_train_le_hold = le.fit_transform(y_train['hold racket handed'])
    y_test_le_hold = le.transform(y_test['hold racket handed'])
    model_binary(X_train_scaled, y_train_le_hold, X_test_scaled, y_test_le_hold)
    print('----------------------------------')

    y_train_le_years = le.fit_transform(y_train['play years'])
    y_test_le_years = le.transform(y_test['play years'])
    model_multiary(X_train_scaled, y_train_le_years, X_test_scaled, y_test_le_years)
    print('----------------------------------')
    
    y_train_le_level = le.fit_transform(y_train['level'])
    y_test_le_level = le.transform(y_test['level'])
    model_multiary(X_train_scaled, y_train_le_level, X_test_scaled, y_test_le_level)
    print('----------------------------------')

    #AUC SCORE: 0.792(gender) + 0.998(hold) + 0.660(years) + 0.822(levels)
    
    # yungan trim
    # Binary F1: 0.4362017804154303
    # Binary ACC: 0.7736842105263158
    # Binary AUC: 0.4510601092896175
    # Binary F1: 0.6880863039399625
    # Binary ACC: 0.8526315789473684
    # Binary AUC: 0.9737916666666667
    # Multiary F1: 0.5134412489038974
    # Multiary ACC: 0.5236842105263158
    # Multiary AUC: 0.6734123961218836
    # Multiary F1: 0.4223753492370514
    # Multiary ACC: 0.6578947368421053
    # Multiary AUC: 0.8394148199445984
    
    # baseline level
    # Binary F1: 0.4422187981510015
    # Binary ACC: 0.7928176795580111
    # Binary AUC: 0.39555130859478677
    # Binary F1: 0.6720753937352002
    # Binary ACC: 0.8425414364640884
    # Binary AUC: 1.0
    # Multiary F1: 0.5986007670876025
    # Multiary ACC: 0.5773480662983426
    # Multiary AUC: 0.6795293947071213
    # Multiary F1: 0.44116503237378835
    # Multiary ACC: 0.6906077348066298
    # Multiary AUC: 0.8440765442650305
    
    # balance level
    # Binary F1: 0.4490106544901065
    # Binary ACC: 0.8149171270718232
    # Binary AUC: 0.4458777937038807
    # Binary F1: 0.6804741189080133
    # Binary ACC: 0.8453038674033149
    # Binary AUC: 0.993626157355638
    # Multiary F1: 0.6041319791319791
    # Multiary ACC: 0.585635359116022
    # Multiary AUC: 0.7030672903757517
    # Multiary F1: 0.447525310863309
    # Multiary ACC: 0.7016574585635359
    # Multiary AUC: 0.8279674511360053
    
    # balance years
    # Binary F1: 0.4833538840937115
    # Binary ACC: 0.9355608591885441
    # Binary AUC: 0.8253023431594859
    # Binary F1: 0.6229869438272398
    # Binary ACC: 0.883054892601432
    # Binary AUC: 0.9994893221912721
    # Multiary F1: 0.35846780328455724
    # Multiary ACC: 0.43914081145584727
    # Multiary AUC: 0.5558794379161659
    # Multiary F1: 0.25296977554493155
    # Multiary ACC: 0.431980906921241
    # Multiary AUC: 0.6822633348712602

    # normal
    # Binary F1: 0.4547945205479452
    # Binary ACC: 0.8341708542713567
    # Binary AUC: 0.8425976633807959
    # Binary F1: 0.9515080358428497
    # Binary ACC: 0.9723618090452262
    # Binary AUC: 0.9991241241241241
    # Multiary F1: 0.530722817534418
    # Multiary ACC: 0.5628140703517588
    # Multiary AUC: 0.666288414433979
    # Multiary F1: 0.391307634164777
    # Multiary ACC: 0.6306532663316583
    # Multiary AUC: 0.8186114828750115
    
    # normal yungan
    # Binary F1: 0.45646437994722955
    # Binary ACC: 0.8398058252427184
    # Binary AUC: 0.7456209493781748
    # Binary F1: 0.8943059505883666
    # Binary ACC: 0.9441747572815534
    # Binary AUC: 0.9779722744360901
    # Multiary F1: 0.48913824571039516
    # Multiary ACC: 0.5339805825242718
    # Multiary AUC: 0.656261782448864
    # Multiary F1: 0.386663634428165
    # Multiary ACC: 0.6140776699029126
    # Multiary AUC: 0.8090683617683099

if __name__ == '__main__':
    main()
