import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import TRAIN_DATA_DIR, TRAIN_INFO, TEST_INFO, TEST_DATA_DIR
from tqdm import tqdm
from alan.gpt import GPT, GPTConfig
from alan.dataloader import TrajectoryDataset
import matplotlib.pyplot as plt
import joblib
import os
import wandb
from copy import deepcopy
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score

def svm_evaluate(model, dataset):
    all_embeddings = []
    all_labels = []
    
    dataloader = DataLoader(dataset, batch_size=128, num_workers=4, shuffle=False, pin_memory=True)
    
    with torch.no_grad():
        for input, _, label in tqdm(dataloader, desc="Extracting SVM embeddings", leave=False):
            input, label_cu = input.to(device), label.to(device)
            if model_args.enable_mode_embedding:
                embedding = model(input, mode=label_cu[:, 0])
            else:
                embedding = model(input)
            embedding = torch.mean(embedding, dim=1).detach().cpu().numpy()
            all_embeddings.append(embedding)
            all_labels.append(label.numpy())
    
    embeddings = np.concatenate(all_embeddings)
    labels = np.concatenate(all_labels)
    
    print(f"all_embeddings shape: {embeddings.shape}")
    print(f"all_labels shape: {labels.shape}")
        
    df = pd.read_csv(TRAIN_INFO)
    unique_players = sorted(df["player_id"].unique())
    
    for task in PREDICTING_FIELDS:
        task_labels = labels[:, PREDICTING_FIELDS.index(task)]
        task_labels = task_labels.astype(int)
        unique_labels = np.unique(task_labels)
        
        all_predictions = []
        all_predictions_probs = []
        all_ground_truths = []
        print(f"Evaluating SVM for task: {task}")
        for valid_player_ids in tqdm(unique_players, desc="Validating players", leave=False):
            # print(f"player: {valid_player_ids}")
            train_player_ids = [p for p in unique_players if p != valid_player_ids]
            df_train = df[df["player_id"].isin(train_player_ids)].index.values
            df_valid = df[df["player_id"] == valid_player_ids].index.values

            train_embeddings = embeddings[df_train]
            train_labels = task_labels[df_train]
            valid_embeddings = embeddings[df_valid]
            valid_labels = task_labels[df_valid]
            
            # Train SVM
            svm = SVC(C=1.0, probability=True, kernel='rbf', class_weight='balanced')
            svm.fit(train_embeddings, train_labels)
            
            # Predict
            predictions = svm.predict(valid_embeddings)
            probabilities = svm.predict_proba(valid_embeddings)
            
            all_predictions.append(predictions)
            all_predictions_probs.append(probabilities)
            all_ground_truths.append(valid_labels)
        
        all_predictions = np.concatenate(all_predictions)
        all_predictions_probs = np.concatenate(all_predictions_probs)
        all_ground_truths = np.concatenate(all_ground_truths)
        
        print(f"Classification Report for {task}:")
        print(classification_report(all_ground_truths, all_predictions))
        f1 = f1_score(all_ground_truths, all_predictions, average='macro')
        accuracy = accuracy_score(all_ground_truths, all_predictions)
        print(f"F1 Score for {task}: {f1:.4f}")
        print(f"Accuracy for {task}: {accuracy:.4f}")
        if len(unique_labels) == 2:
            micro_roc_auc = roc_auc_score(all_ground_truths, all_predictions_probs[:, 1], average='micro')
            macro_roc_auc = roc_auc_score(all_ground_truths, all_predictions_probs[:, 1], average='macro')
            
            print(f"Micro ROC AUC score: {micro_roc_auc:.4f}")
            print(f"Macro ROC AUC score: {macro_roc_auc:.4f}")
        else:
            micro_roc_auc = roc_auc_score(all_ground_truths, all_predictions_probs, average='micro', multi_class='ovr')
            macro_roc_auc = roc_auc_score(all_ground_truths, all_predictions_probs, average='macro', multi_class='ovr')

            print(f"Micro ROC AUC score: {micro_roc_auc:.4f}")
            print(f"Macro ROC AUC score: {macro_roc_auc:.4f}")
            
        if use_wandb:
            wandb.log({
                f"{task}_f1_score": f1,
                f"{task}_accuracy": accuracy,
                f"{task}_micro_roc_auc": micro_roc_auc,
                f"{task}_macro_roc_auc": macro_roc_auc,
            })



device = "cuda:0"
model_args = GPTConfig(
    max_seq_len = 1024,
    in_chans = 16,
    n_layer = 8,
    n_head = 8,
    n_embd = 128,
    ff = 4,
    patch_size = 8,
    dropout = 0.2,
    bias = False,
    enable_mode_embedding = False,
)

batch_size = 32
learning_rate = 5e-3
weight_decay = 0.001
betas = (0.9, 0.95)
num_epochs = 5000

use_scaler = True
detrend_and_filter = True
data_transform = "diff_and_magnitude"
filter_order = 4
freq_cutoff = 30  # Hz
out_dir = "outs/out61"
use_wandb = True

PREDICTING_FIELDS = [
    "mode",
    "gender",
    "hold racket handed",
    "play years",
    "level",
]

TEST_PREDICTING_FIELDS = [
    "mode",
]

os.makedirs(out_dir, exist_ok=True)

print(f"preparing dataset...")
train_df = pd.read_csv(TRAIN_INFO)
test_df = pd.read_csv(TEST_INFO)

train_dataset = TrajectoryDataset(
    TRAIN_DATA_DIR,
    train_df,
    train=True,
    max_seq_len=model_args.max_seq_len,
    patch_size=model_args.patch_size,
    use_scaler=use_scaler,
    detrend_and_filter=detrend_and_filter,
    label=True,
    predicting_fields=PREDICTING_FIELDS,
    filter_order=filter_order,
    freq_cutoff=freq_cutoff,
    data_transform=data_transform,
)

if use_scaler:
    joblib.dump(train_dataset.scaler, f"{out_dir}/scaler.joblib")

valid_dataset = TrajectoryDataset(
    TEST_DATA_DIR,
    test_df,
    train=True,
    max_seq_len=model_args.max_seq_len,
    patch_size=model_args.patch_size,
    use_scaler=use_scaler,
    scaler=train_dataset.scaler if use_scaler else None,
    encoder=train_dataset.encoders,
    sample_weights=False,
    detrend_and_filter=detrend_and_filter,
    label=True,
    predicting_fields=TEST_PREDICTING_FIELDS,
    filter_order=filter_order,
    freq_cutoff=freq_cutoff,
    data_transform=data_transform,
)

svm_dataset = TrajectoryDataset(
    TRAIN_DATA_DIR,
    train_df,
    train=True,
    max_seq_len=model_args.max_seq_len,
    patch_size=model_args.patch_size,
    use_scaler=use_scaler,
    scaler=train_dataset.scaler if use_scaler else None,
    encoder=train_dataset.encoders,
    sample_weights=False,
    detrend_and_filter=detrend_and_filter,
    label=True,
    predicting_fields=PREDICTING_FIELDS,
    filter_order=filter_order,
    freq_cutoff=freq_cutoff,
    data_transform=data_transform,
)

print(f"train dataset size: {len(train_dataset)}")
print(f"valid dataset size: {len(valid_dataset)}")

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

input, target, label = next(iter(valid_dataloader))
print(f"input shape: {input.shape}")
print(f"target shape: {target.shape}")
print(f"label shape: {label.shape}")
# exit(0)

model = GPT(model_args)
optimizer = model.configure_optimizers(learning_rate=learning_rate, weight_decay=weight_decay, betas=betas, device_type=device)
model = model.to(device)

if use_wandb:
    wandb_configs = {
        "n_embd": model_args.n_embd,
        "n_layer": model_args.n_layer,
        "n_head": model_args.n_head,
        "in_chans": model_args.in_chans,
        "patch_size": model_args.patch_size,
        "max_seq_len": model_args.max_seq_len,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "betas": betas,
        "num_epochs": num_epochs,
        "use_scaler": use_scaler,
        "out_dir": out_dir,
        "cutoff": freq_cutoff,
        "predicting_fields": PREDICTING_FIELDS,
        "detrend_and_filter": detrend_and_filter,
        "enable_mode_embedding": model_args.enable_mode_embedding,
        "ff": model_args.ff,
        "filter_order": filter_order,
        "mode_in_class_weights": True,
    }

    wandb.init(project="imugpt-experiments-002", config=wandb_configs)

best_valid_loss = float('inf')
epochs_since_improvement = 0
for epoch in range(num_epochs):
    total_train_loss = 0.0
    total_valid_loss = 0.0
    train_seen_items = 0
    val_seen_items = 0
    
    model.train()
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    for i, (input, target, label) in enumerate(pbar):
        input, target, label_cu = input.to(device), target.to(device), label.to(device)
        optimizer.zero_grad()
        if model_args.enable_mode_embedding:
            logits, loss, embedding = model(input, target, label_cu[:, 0])
        else:
            logits, loss, embedding = model(input, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_train_loss += loss.item()
        train_seen_items += input.size(0)
    
    total_train_loss /= train_seen_items
    if use_wandb:
        wandb.log({
            "train_loss": total_train_loss,
        })
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {total_train_loss:.4f}")

    model.eval()
    with torch.no_grad():
        for _ in range(30):
            pbar = tqdm(valid_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
            for i, (input, target, label) in enumerate(pbar):
                input, target, label_cu = input.to(device), target.to(device), label.to(device)
                if model_args.enable_mode_embedding:
                    logits, loss, embedding = model(input, target, label_cu[:, 0])
                else:
                    logits, loss, embedding = model(input, target)
                total_valid_loss += loss.item()
                val_seen_items += input.size(0)
                
    total_valid_loss /= val_seen_items
    if use_wandb:
        wandb.log({
            "valid_loss": total_valid_loss,
        })

    print(f"Epoch {epoch+1}/{num_epochs} - Valid Loss: {total_valid_loss:.4f}")
    
    if (epoch + 1) % 40 == 0 or epoch == 0:
        print(f"Saving model at epoch {epoch+1}...")
        torch.save(model.state_dict(), f"{out_dir}/model_epoch_{epoch+1}.pth")
        svm_evaluate(model, svm_dataset)
    
    if total_valid_loss < best_valid_loss:
        best_valid_loss = total_valid_loss
        best_model_wts = deepcopy(model.state_dict())
        epochs_since_improvement = 0
    else:
        epochs_since_improvement += 1
        if epochs_since_improvement > 42:
            print(f"Early stopping triggered after {epochs_since_improvement} epochs without improvement.")
            break
        print(f"No improvement in validation loss for {epochs_since_improvement} epochs.")
    
# Save the best model weights
torch.save(best_model_wts, f"{out_dir}/best_model.pth")
torch.save(model.state_dict(), f"{out_dir}/model_final.pth")

svm_evaluate(model, svm_dataset)

if use_wandb:
    wandb.finish()

