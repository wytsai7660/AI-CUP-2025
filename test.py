# test.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import argparse
import time
from tqdm import tqdm
import numpy as np
import pandas as pd # 用於保存結果

# 從 model.py 導入模型構建函數（假設模型代碼在 model.py）
try:
    from model import build_transformer
except ImportError:
    print("錯誤：無法從 model.py 導入 build_transformer。請確保 model.py 存在且包含該函數。")
    exit(1)

# 從 dataloader.py 導入測試數據集和 collate function
try:
    from dataloader import TestTimeSeriesDataset, universal_collate_fn
except ImportError:
     print("錯誤：無法從 dataloader.py 導入 TestTimeSeriesDataset 或 universal_collate_fn。請確保 dataloader.py 存在。")
     exit(1)


# --- 輔助函數 (與 train.py 類似) ---
def create_masks(src_batch, tgt_batch, pad_idx=0.0):
    """創建 Transformer 所需的 masks。"""
    src_mask = (src_batch[:, :, 0] != pad_idx).unsqueeze(1).unsqueeze(2)
    tgt_seq_len = tgt_batch.size(1)
    tgt_pad_mask = (tgt_batch[:, :, 0] != pad_idx).unsqueeze(1).unsqueeze(-1) # (batch, 1, tgt_len, 1)
    look_ahead_mask = torch.tril(torch.ones((tgt_seq_len, tgt_seq_len), device=tgt_batch.device)).bool() # (tgt_len, tgt_len)
    tgt_mask = tgt_pad_mask & look_ahead_mask # (batch, 1, tgt_len, tgt_len)
    return src_mask, tgt_mask

# --- 主執行邏輯 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Transformer for Time Series Data")

    # 必要的參數
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the trained model checkpoint (.pth.tar file)')
    parser.add_argument('--test_data_dir', type=str, default='./now dir/39_Test_Dataset/test_data', help='Directory containing the test .txt files')
    parser.add_argument('--output_csv', type=str, default='./test_predictions.csv', help='Path to save the prediction results CSV file')

    # 可選參數
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader')
    parser.add_argument('--device', type=str, default=None, help='Device to use (e.g., "cuda", "cpu"). Auto-detects if None.')

    args = parser.parse_args()

    # --- 設備設置 ---
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 加載 Checkpoint ---
    if not os.path.isfile(args.checkpoint):
        print(f"錯誤：找不到 Checkpoint 文件：{args.checkpoint}")
        exit(1)

    print(f"Loading checkpoint: {args.checkpoint}")
    # 加載到 CPU 以避免 GPU 內存問題，稍後再移到目標 device
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    # 從 checkpoint 中獲取模型參數 (如果保存了的話)
    if 'args' in checkpoint:
        model_args = checkpoint['args']
        print("Loaded model hyperparameters from checkpoint.")
    else:
        # 如果 checkpoint 中沒有保存參數，需要手動提供或從腳本參數讀取
        print("警告：Checkpoint 中未找到模型參數 ('args')。將使用默認值或命令行參數。")
        print("請確保測試時使用的模型參數與訓練時一致！")
        # 在這種情況下，你需要確保 build_transformer 使用的參數是正確的
        # 例如，從 argparse 獲取 d_model, n_heads 等，或者硬編碼
        # 這是不推薦的做法
        parser.add_argument('--d_input', type=int, default=6)
        parser.add_argument('--output_dim', type=int, default=2)
        parser.add_argument('--d_model', type=int, default=32)
        parser.add_argument('--d_ff', type=int, default=128)
        parser.add_argument('--n_layers', type=int, default=4)
        parser.add_argument('--n_heads', type=int, default=4)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--max_seq_len', type=int, default=2000)
        # 重新解析參數以包含模型參數（僅在 checkpoint 中未找到 'args' 時）
        args = parser.parse_args()
        model_args = args


    # --- 構建模型 ---
    print("Building model...")
    try:
        model = build_transformer(
            d_input=model_args.d_input,
            output_dim=model_args.output_dim,
            N=model_args.n_layers,
            d_model=model_args.d_model,
            d_ff=model_args.d_ff,
            h=model_args.n_heads,
            dropout=model_args.dropout, # 在 eval 模式下 dropout 通常不生效，但保持一致性
            max_seq_len=model_args.max_seq_len
        )
    except AttributeError as e:
         print(f"錯誤：無法從加載的參數構建模型。缺少參數：{e}")
         print("請確保 Checkpoint 中的 'args' 對象包含所有需要的模型超參數，")
         print("或者在命令行中提供它們（如果 Checkpoint 中沒有 'args'）。")
         exit(1)


    # 加載模型狀態字典
    try:
        model.load_state_dict(checkpoint['state_dict'])
    except KeyError:
         print("錯誤：Checkpoint 中未找到 'state_dict'。")
         exit(1)
    except RuntimeError as e:
         print(f"錯誤：加載 state_dict 時出錯（可能模型結構不匹配）：{e}")
         exit(1)

    model = model.to(device)
    model.eval() # 設置為評估模式！
    print("Model loaded and set to evaluation mode.")

    # --- 加載測試數據 ---
    print(f"Loading test data from: {args.test_data_dir}")
    try:
        test_dataset = TestTimeSeriesDataset(test_data_dir=args.test_data_dir)
        if len(test_dataset) == 0:
            print("錯誤：測試數據集為空。請檢查測試數據目錄。")
            exit(1)

        test_dataloader = DataLoader(test_dataset,
                                     batch_size=args.batch_size,
                                     shuffle=False, # 測試時不需要打亂
                                     collate_fn=universal_collate_fn, # 使用通用 collate function
                                     num_workers=args.num_workers)
        print(f"Test data loaded: {len(test_dataset)} samples.")
    except FileNotFoundError as e:
         print(f"錯誤：找不到測試數據目錄：{e}")
         exit(1)
    except Exception as e:
         print(f"加載測試數據時發生錯誤：{e}")
         exit(1)


    # --- 執行預測 ---
    predictions = {} # 使用字典存儲結果，鍵為文件名
    print("Starting prediction...")
    start_time = time.time()

    with torch.no_grad(): # 在評估時不需要計算梯度
        pbar = tqdm(test_dataloader, desc="Predicting")
        for batch in pbar:
            padded_time_series, _, lengths, filenames = batch # _ 接收 metadata (在此為 None)

            if padded_time_series is None or lengths is None or filenames is None:
                print("警告：DataLoader 返回了無效的批次，跳過。")
                continue

            padded_time_series = padded_time_series.to(device)

            # --- 準備模型輸入 (與驗證時類似) ---
            # 如果序列長度為 1 或 0，無法進行移位預測
            if padded_time_series.size(1) <= 1:
                # print(f"警告：批次中的序列長度過短 ({padded_time_series.size(1)})，無法生成 tgt_input。")
                # 為這些文件生成空預測或 NaN
                for i, fname in enumerate(filenames):
                     predictions[fname] = np.array([]) # 或者 [[np.nan, np.nan]]
                continue


            src = padded_time_series
            # 解碼器輸入：使用除最後一個時間步之外的所有數據
            tgt_input = padded_time_series[:, :-1, :]

            # 創建 Masks
            src_mask, tgt_mask = create_masks(src, tgt_input, pad_idx=0.0)

            # --- 模型前向傳播 ---
            # output shape: (batch_size, seq_len-1, output_dim)
            output = model(src, tgt_input, src_mask, tgt_mask)

            # 將預測結果轉移回 CPU 並轉換為 numpy
            output_np = output.cpu().numpy()

            # --- 處理批次中的每個樣本 ---
            for i in range(output_np.shape[0]):
                filename = filenames[i]
                original_length = lengths[i].item() # 獲取原始長度
                # 預測的有效長度是 original_length - 1
                # （因為我們預測從第二個時間步開始的序列）
                valid_prediction_length = max(0, original_length - 1)

                if valid_prediction_length > 0:
                    # 提取有效的預測部分
                    # output_np[i] shape: (seq_len-1, output_dim)
                    sample_prediction = output_np[i, :valid_prediction_length, :]
                else:
                    # 如果原始長度為 1 或 0，則沒有有效的預測
                    sample_prediction = np.array([]) # 或者 [[np.nan, np.nan]]

                predictions[filename] = sample_prediction

    end_time = time.time()
    print(f"Prediction finished in {end_time - start_time:.2f} seconds.")

    # --- 保存結果 ---
    print(f"Saving predictions to {args.output_csv}...")
    try:
        # 將字典轉換為更易於保存的格式，例如 Pandas DataFrame
        results_list = []
        for filename, pred_array in predictions.items():
            # 對於每個時間步的預測 (假設 output_dim=2)
            for step, (pred_val1, pred_val2) in enumerate(pred_array):
                 results_list.append({
                     'filename': filename,
                     'timestep': step, # 預測的是原始數據的 timestep+1
                     'prediction_1': pred_val1,
                     'prediction_2': pred_val2
                 })
            # 如果預測為空（原始序列太短）
            if len(pred_array) == 0:
                 results_list.append({
                     'filename': filename,
                     'timestep': -1, # 特殊值表示無預測
                     'prediction_1': np.nan,
                     'prediction_2': np.nan
                 })


        results_df = pd.DataFrame(results_list)
        results_df.to_csv(args.output_csv, index=False)
        print("Predictions saved successfully.")
    except Exception as e:
        print(f"保存預測結果時出錯: {e}")
        # 可以嘗試將原始字典保存為 numpy 文件作為備份
        try:
             np.savez_compressed("test_predictions_backup.npz", **predictions)
             print("已將原始預測保存到 test_predictions_backup.npz")
        except Exception as e_np:
             print(f"備份預測到 npz 文件也失敗: {e_np}")