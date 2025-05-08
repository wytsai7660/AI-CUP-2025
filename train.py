import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from model import EncoderOnlyClassifier
from helper.dataloader import SwingDataset, collate_fn_torch

import os
import copy
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import time
import multiprocessing

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def normalize(x):
    return (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)
def main():
# training parameters
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    epochs = 20
    learning_rate = 0.0001
    train_bathch_size = 32
    valid_batch_size = 32

    # data path and weight path
    base_path = os.path.dirname(os.path.abspath(__file__))
    train_data_path = os.path.join(base_path, "39_Training_Dataset", "train_data") 
    label_data_path = os.path.join(base_path, "39_Training_Dataset", "train_info.csv")
    result_path = os.path.join(base_path, "result")

    s = time.strftime("%m%d%H%M", time.localtime())
    filename = f"weight_{s}.pth"
    weight_path = os.path.join(base_path, "weights", filename)

    #dataloader
    ds = SwingDataset(
            data_dir=train_data_path,
            info_csv=label_data_path,
            smooth_w=5,
            perc=75,
            dist_frac=0.3,
            min_len=20,
            max_len=500,
            transform=normalize
        ) 
    total_len = len(ds)
    train_len = int(0.8 * total_len)
    valid_len = total_len - train_len

    train_ds, valid_ds = random_split(
        ds,
        [train_len, valid_len],
        generator=torch.Generator().manual_seed(42)  # 固定隨機種子，方便重現
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_bathch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn_torch
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=valid_batch_size,
        shuffle=False,        # valid 通常不需要 shuffle
        num_workers=4,
        collate_fn=collate_fn_torch,
        drop_last=True  
    )
    # model
    model = EncoderOnlyClassifier(d_model=6, n_enc=3, dim_ff=512)
    model = model.to(device)

    # set optimizer and loss function

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # train
    train_loss_list = list()
    valid_loss_list = list()
    train_accuracy_list = list()
    valid_accuracy_list = list()
    best = 100
    best_model_wts = copy.deepcopy(model.state_dict())

    patience = 8
    no_improve = 0
    
    # for batch in train_loader:
    #     print([t.shape for t in batch])
    #     break


    for epoch in range(epochs):
        print(f'\nEpoch: {epoch+1}/{epochs}')
        print('-' * len(f'Epoch: {epoch+1}/{epochs}'))
        train_loss, valid_loss = 0.0, 0.0
        train_accuracy, valid_accuracy = 0.0, 0.0
        train_correct, valid_correct = 0, 0
        train_samples = 0
        valid_samples = 0

        model.train()
        for src, tgt, label_onehot in tqdm(train_loader, desc="Training"):
            src, tgt, label_onehot = src.to(device), tgt.to(device), label_onehot.to(device)

            # src, tgt: (batch_size, seq_len, 6)
            src = src.permute(1, 0, 2) # => (seq_len, batch_size, 6) 
            optimizer.zero_grad()
            # forward + backward + optimize
            output = model(src)
            
            loss = 0
            bsz = output.size(0)
            for (start, end) in [(0,2), (2,4), (4,7), (7,11)]:
                logits_group = output[:, start:end]                # [B, group_size]
                targets_group = label_onehot[:, start:end].argmax(dim=1)  # [B]
                loss += criterion(logits_group, targets_group)     # 累加每组 loss

                preds_group = logits_group.argmax(dim=1)           # [B]
                train_correct += (preds_group == targets_group).sum().item()
                train_samples += bsz

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * bsz

        train_loss /= train_samples
        train_loss_list.append(train_loss)
        train_accuracy = float(train_correct) / train_samples
        train_accuracy_list.append((train_accuracy))

        model.eval()
        with torch.no_grad():
            for src, _, label_onehot in tqdm(valid_loader, desc="Validating"):
                # 1) 移到 GPU 并 permute
                src          = src.to(device).permute(1, 0, 2)       # [L, B, 6]
                label_onehot = label_onehot.to(device)              # [B, 11]

                # 2) Forward
                output = model(src)                            # [B, 11]

                # 3) 对每个子段，动态读取当前 batch size
                for start, end in [(0,2), (2,4), (4,7), (7,11)]:
                    logits_group  = output[:, start:end]            # [cur_B, group_size]
                    targets_group = label_onehot[:, start:end].argmax(dim=1)  # [cur_B]
                    cur_B = logits_group.size(0)                    # 真正的 batch 大小

                    # 累加加权 loss
                    valid_loss    += criterion(logits_group, targets_group).item() * cur_B
                    # 累加正确数
                    valid_correct += (logits_group.argmax(dim=1) == targets_group).sum().item()
                    # 累加样本数（按四段算，所以会加四次 cur_B）
                    valid_samples += cur_B
            valid_loss /= valid_samples
            valid_loss_list.append(valid_loss)
            valid_accuracy = float(valid_correct) / valid_samples
            valid_accuracy_list.append((valid_accuracy))
        
        # print loss and accuracy in one epoch
        print(f'Training loss: {train_loss:.4f}, validation loss: {valid_loss:.4f}')
        print(f'Training accuracy: {train_accuracy:.4f}, validation accuracy: {valid_accuracy:.4f}')

        # record best weight so far
        if valid_loss < best :
            best = valid_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        scheduler.step()

    torch.save(best_model_wts, weight_path)

    # plot the loss curve for training and validation
    print("\nFinished Training")
    pd.DataFrame({
        "train-loss": train_loss_list,
        "valid-loss": valid_loss_list
    }).plot()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlim(1,epoch+1)
    plt.xlabel("Epoch"),plt.ylabel("Loss")
    plt.savefig(os.path.join(result_path, "result", "Loss_curve"))

    # plot the accuracy curve for training and validation
    pd.DataFrame({
        "train-accuracy": train_accuracy_list,
        "valid-accuracy": valid_accuracy_list
    }).plot()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlim(1,epoch+1)
    plt.xlabel("Epoch"),plt.ylabel("Accuracy")
    plt.savefig(os.path.join(result_path, "result", "Training_accuracy"))
    
    
if __name__ == "__main__":
    multiprocessing.freeze_support()  # Windows 下多进程需要
    main()