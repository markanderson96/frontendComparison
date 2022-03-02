import torch
import torch.nn as nn
import torchaudio

from torch import Tensor

class SpecAugment(nn.Module):
    def __init__(self,
                 time_mask_param: int,
                 freq_mask_param: int,
                 n_fft: int,
                 rate: float,
                 time_iid: bool = True,
                 freq_iid: bool = True
    ) -> None:
        super(SpecAugment, self).__init__()

        self.time_mask = torchaudio.transforms.TimeMasking(
            time_mask_param=time_mask_param,
            iid_masks=time_iid
        )
        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=freq_mask_param,
            iid_masks=freq_iid
        )
        self.time_stretch = torchaudio.transforms.TimeStretch(
            n_freq=n_fft//2 + 1,
            fixed_rate=rate
        )
    
    def forward(self, x) -> Tensor:
        rand = torch.rand(3)
        if rand[2] < 0.25:
            x = self.time_stretch(x)
        x = x.abs()
        if rand[0] < 0.25:
            x = self.time_mask(x)
        if rand[1] < 0.25:
            x = self.freq_mask(x)

        return x