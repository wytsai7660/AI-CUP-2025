from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from config import (
    COUNTS,
    DEVICE,
    FIELD_LENS,
    MEAN,
    PREDICTING_FIELDS,
    STD,
    TRAIN_DATA_DIR,
    TRAIN_INFO,
)
from helper.dataloader import get_train_valid_dataloader
from helper.segment import Trim, Yungan
from helper.transform import Compose, Normalize, RandomCrop
from wytsai7660.model import MultiTaskTransformerEncoder


USE_WANDB = True

if USE_WANDB:
    wandb_config = SimpleNamespace(
        split_type="level",
        model_type="encoder",
    )

    wandb.init(
        entity="ai-cup-2025",
        project="ur_mom_fat",
        config=wandb_config,
    )


def metric_report(ground_truth: torch.Tensor, prediction: torch.Tensor, field: str):
    """
    tensor (batch_size, class_num)
    """
    y_true = ground_truth.detach().cpu().numpy()
    y_pred = prediction.detach().cpu().numpy()
    # print(y_true.shape, y_pred.shape)

    y_true = np.argmax(y_true, axis=1)
    y_pred = torch.softmax(torch.tensor(y_pred), dim=1).numpy()
    # print(np.unique(y_true))
    # print(y_pred)
    if y_pred.shape[1] == 2:
        micro_roc_score = roc_auc_score(
            y_true,
            y_pred[:, 1],
            labels=np.arange(FIELD_LENS[PREDICTING_FIELDS.index(field)]),
        )
        macro_roc_score = roc_auc_score(
            y_true,
            y_pred[:, 1],
            labels=np.arange(FIELD_LENS[PREDICTING_FIELDS.index(field)]),
        )
    else:
        micro_roc_score = roc_auc_score(
            y_true,
            y_pred,
            average="micro",
            multi_class="ovr",
            labels=np.arange(FIELD_LENS[PREDICTING_FIELDS.index(field)]),
        )
        macro_roc_score = roc_auc_score(
            y_true,
            y_pred,
            average="macro",
            multi_class="ovr",
            labels=np.arange(FIELD_LENS[PREDICTING_FIELDS.index(field)]),
        )

    y_pred = np.argmax(y_pred, axis=1)
    acc_score = accuracy_score(y_true, y_pred)
    micro_f1_score = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1_score = f1_score(y_true, y_pred, average="macro", zero_division=0)

    metrics = {
        f"{field} micro_roc": micro_roc_score,
        f"{field} macro_roc": macro_roc_score,
        f"{field} acc": acc_score,
        f"{field} micro_f1": micro_f1_score,
        f"{field} macro_f1": macro_f1_score,
    }
    return metrics


DROPOUT = 0.2
MAX_SEQ_LEN = 512
LEARNING_RATE = 0.0005
WARMUP_EPOCHS = 10
EPOCHS = 100


model = MultiTaskTransformerEncoder(dropout=DROPOUT, max_len=MAX_SEQ_LEN).to(DEVICE)
print(model)

train_loader, valid_loader = get_train_valid_dataloader(
    TRAIN_DATA_DIR,
    TRAIN_INFO,
    split_target="level",
    batch_size=8,
    # segment=Trim(),
    segment=Yungan(),
    # segment=None,
    max_duration=MAX_SEQ_LEN,
    # train_transform=Compose([RandomCrop((64, 512)), Normalize(mean=MEAN, std=STD)]),
    # valid_transform=Compose([RandomCrop((64, 512)), Normalize(mean=MEAN, std=STD)]),
    train_transform=Compose([Normalize(mean=MEAN, std=STD)]),
    valid_transform=Compose([Normalize(mean=MEAN, std=STD)]),
)

criterions = [
    nn.CrossEntropyLoss(weight=(1 / torch.tensor(count)).to(DEVICE)) for count in COUNTS
]

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

scheduler = optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[
        optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=max(1, WARMUP_EPOCHS)
        ),
        optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, EPOCHS - WARMUP_EPOCHS),
            eta_min=LEARNING_RATE / 10,
        ),
    ],
    milestones=[WARMUP_EPOCHS] if WARMUP_EPOCHS > 0 else [1],
)

LOSS_WEIGHTS = [1.0, 1.0, 2.0, 2.0]


print("Starting Training Loop...")
for epoch in range(EPOCHS):
    # Training
    model.train()
    for inputs, lengths, metas in tqdm(train_loader, desc=f"{epoch}", leave=False):
        inputs: torch.Tensor = inputs.to(DEVICE)
        lengths: torch.Tensor = lengths.to(DEVICE)
        metas: list[torch.Tensor] = [meta.to(DEVICE) for meta in metas]

        outputs = model(inputs, lengths)

        full_loss = torch.tensor(0.0, device=DEVICE)
        for field, output, meta, criterion, w in zip(
            PREDICTING_FIELDS, outputs, metas, criterions, LOSS_WEIGHTS
        ):
            loss = criterion(output, meta.argmax(dim=-1))
            full_loss += w * loss

        optimizer.zero_grad()
        full_loss.backward()
        optimizer.step()

    # Validation
    from collections import defaultdict

    all_metas, all_outputs = defaultdict(list), defaultdict(list)
    valid_loss = 0.0
    model.eval()

    with torch.no_grad():
        for inputs, lengths, metas in tqdm(valid_loader, desc=f"{epoch}", leave=False):
            inputs = inputs.to(DEVICE)
            lengths = lengths.to(DEVICE)
            metas: list[torch.Tensor] = [meta.to(DEVICE) for meta in metas]

            outputs = model(inputs, lengths)

            for field, output, meta, criterion, w in zip(
                PREDICTING_FIELDS, outputs, metas, criterions, LOSS_WEIGHTS
            ):
                loss = criterion(output, meta.argmax(dim=-1))
                valid_loss += w * loss.item()
                all_metas[field].append(meta.detach().cpu())
                all_outputs[field].append(output.detach().cpu())

    if USE_WANDB:
        metrics = {}
        for field in PREDICTING_FIELDS:
            meta = torch.cat(all_metas[field], dim=0)
            output = torch.cat(all_outputs[field], dim=0)
            metrics |= metric_report(meta, output, field)
        metrics["valid_loss"] = valid_loss
        wandb.log(metrics)

    scheduler.step()

if USE_WANDB:
    wandb.finish()
