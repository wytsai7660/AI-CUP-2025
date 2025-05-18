import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

# ===== Paths =====
BASE_PATH = Path(__file__).parent

TRAIN_DATA_DIR = BASE_PATH / "39_Training_Dataset" / "train_data"
TRAIN_INFO = BASE_PATH / "39_Training_Dataset" / "train_info.csv"

TEST_DATA_DIR = BASE_PATH / "39_Test_Dataset" / "test_data"
TEST_INFO = BASE_PATH / "39_Test_Dataset" / "test_info.csv"

# Check if any of the paths are absent
for path in [TRAIN_DATA_DIR, TRAIN_INFO, TEST_DATA_DIR, TEST_INFO]:
    if not path.exists():
        sys.exit(f"{path}: No such file or directory")


# ===== Constants =====
PREDICTING_FIELDS = [
    "gender",
    "hold racket handed",
    "play years",
    "level",
]

POSSIBLE_VALUES = [
    [1, 2],
    [1, 2],
    [0, 1, 2],
    [2, 3, 4, 5],
]

NUM_WORKERS = 4


# ===== Reproducibility =====

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# Set to True for max reproducibility, but performance will be much slower
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
