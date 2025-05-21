import pandas as pd    
import torch
from config import TRAIN_INFO

def make_weights(counts):
    n = counts.sum()
    c = len(counts)
    # 根据 pandas 索引顺序取值
    return [(n/(c * counts[i])) for i in counts.index]

def class_weights(csvpath,device):
    df = pd.read_csv(csvpath)
    counts_gender = df['gender'].value_counts().sort_index()               # index=[1,2]
    counts_hold   = df['hold racket handed'].value_counts().sort_index()  # index=[1,2]
    counts_play   = df['play years'].value_counts().sort_index()          # index=[0,1,2]
    counts_level  = df['level'].value_counts().sort_index()               # index=[2,3,4,5]
    # print(counts_gender)
    # print(counts_hold)
    # print(counts_play)
    # print(counts_level)
    
    w_gender = make_weights(counts_gender)   # len=2
    w_hold   = make_weights(counts_hold)     # len=2
    w_play   = make_weights(counts_play)     # len=3
    w_level  = make_weights(counts_level)    # len=4
    class_weights = torch.tensor(
        w_gender + w_hold + w_play + w_level,
        dtype=torch.float,
        device=device
    )
    return class_weights
    
if __name__ == "__main__":
    csvpath = TRAIN_INFO
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = class_weights(csvpath,device)
    # print(class_weights)