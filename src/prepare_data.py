import os
from tkinter import W
import librosa
import datasets
import pandas as pd
import numpy as np
import datasets

import torch
from torch.utils.data import Dataset

from typing import Tuple

class birdDataset(Dataset):
    def __init__(
        self,
        df,
        root_data_dir,
        sample_rate
    ):
        self.meta = df
        self.root_data_dir = root_data_dir
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        audio_path = os.path.join(
            self.root_data_dir,
            str(self.meta.iloc[idx, 1]),
            'wav',
            str(self.meta.iloc[idx, 0])
        )

        y, sr = librosa.load(
            audio_path + '.wav',
            sr=self.sample_rate,
            mono=True,
            res_type='kaiser_fast'
        )

        y = librosa.util.fix_length(y, 441000)

        label = self.meta.iloc[idx, 2]

        return torch.tensor(y), torch.tensor(label)
