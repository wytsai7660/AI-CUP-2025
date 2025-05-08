import os
import torch
import numpy as np
import csv
from model import EncoderOnlyClassifier      # 請確認你的 model 類名
from helper.cut import segment_file           # 你的切片函式
from tqdm import tqdm

# ─── 在這裡直接定義變數 ────────────────────────────────────────────
TEST_DIR    = "39_Test_Dataset/test_data"       # 測試資料夾路徑
WEIGHT_PATH = "weight/weight_05082319.pth"      # 訓練好權重的 .pth 檔
OUTPUT_CSV  = "test_results.csv"                # 輸出 CSV 檔名
# ───────────────────────────────────────────────────────────────────
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def normalize(x: torch.Tensor) -> torch.Tensor:
    # 同訓練時相同的正規化
    return (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 建立並載入模型
    model = EncoderOnlyClassifier(d_model=6, n_enc=3, dim_ff=512).to(device)
    state = torch.load(WEIGHT_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 2) 讀取所有 .txt 檔案，逐一推論
    results = []
    for fn in tqdm(sorted(os.listdir(TEST_DIR))):
        if not fn.lower().endswith(".txt"):
            continue

        unique_id = os.path.splitext(fn)[0]
        path = os.path.join(TEST_DIR, fn)
        data = np.loadtxt(path)            # shape [seq_len, 6]

        # 2.1) 切割：segment_file 返回 (df, segments)
        _, segments = segment_file(path)
        if not segments:
            continue

        # 2.2) 對每個片段做推論並收集概率
        p0_list, p1_list = [], []
        p2_list, p3_list = [], []
        with torch.no_grad():
            for start, length in segments:
                seg = data[start:start + length]           # [length, 6]
                x = torch.from_numpy(seg).float().to(device)
                x = normalize(x)
                src = x.unsqueeze(1)                      # [length, 1, 6]

                # Encoder-only forward
                output = model(src)                       # [1, num_classes]
                logits = output.squeeze(0)                # [num_classes]

                # 四段 softmax
                p0_list.append(torch.softmax(logits[0:2],  dim=0)[0].item())
                p1_list.append(torch.softmax(logits[2:4],  dim=0)[0].item())
                p2_list.append(torch.softmax(logits[4:7],  dim=0).cpu().tolist())
                p3_list.append(torch.softmax(logits[7:11], dim=0).cpu().tolist())

        # 2.3) 將各段機率取平均
        p0 = sum(p0_list) / len(p0_list)
        p1 = sum(p1_list) / len(p1_list)
        p2 = [sum(col) / len(p2_list) for col in zip(*p2_list)]
        p3 = [sum(col) / len(p3_list) for col in zip(*p3_list)]

        # 2.4) 格式化四位小數並加入結果
        row = [unique_id,
               f"{p0:.4f}",
               f"{p1:.4f}",
               *(f"{v:.4f}" for v in p2),
               *(f"{v:.4f}" for v in p3)]
        results.append(row)

    # 3) 輸出 CSV
    header = [
        "unique_id",
        "gender",
        "hold racket handed",
        "play years_0","play years_1","play years_2",
        "level_2","level_3","level_4","level_5"
    ]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(results)

    print(f"Saved results to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()