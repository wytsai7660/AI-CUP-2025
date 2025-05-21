# helper/lr_scheduler.py
import math
from torch.optim.lr_scheduler import LambdaLR


def get_linear_warmup_cosine_scheduler(optimizer, epochs, train_loader_len, warmup_ratio=0.1):
    """
    返回一个线性 Warmup + Cosine Decay 的学习率调度器。
    :param optimizer: 需要调度的 optimizer
    :param epochs: 总训练 epoch 数
    :param train_loader_len: 每 epoch 的 batch 数
    :param warmup_ratio: warmup 占总步数的比例
    :return: LambdaLR 调度器
    """
    total_steps = epochs * train_loader_len
    warmup_steps = int(warmup_ratio * total_steps)

    def lr_lambda(current_step):
        # 线性 warmup
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # cosine decay
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)