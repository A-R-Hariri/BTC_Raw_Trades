import os
import warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")

from dataset import *
from train import *


if __name__ == "__main__":
    seed_all(SEED)

    if MODE == "prepare":
        prepare_dataset()

    elif MODE == "train":
        train()