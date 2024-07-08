import os
import torch as th
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

CUDA_AVALABLE = th.cuda.is_available()
GPUS = 1 if CUDA_AVALABLE else 0
DEVICE = th.device("cuda" if CUDA_AVALABLE else "cpu")

PROJECT_ROOT = Path(os.path.abspath(__file__)).parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "pre_models"
ATK_MODELS_DIR = PROJECT_ROOT / "atk_models"

DATETIME_FMT = "%Y-%m-%d_%H-%M-%S"

EPSILON = 1E-9
