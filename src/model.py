import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import torchaudio
import torchvision
import timm

from torch.utils.tensorboard import SummaryWriter
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from typing import Any, Dict, Tuple
from torch import Tensor

from losses import BCELossModified
from modules import ConvBlock1D, ConvBlock2D, AttnBlock, init_layer
from specaugment import SpecAugment

class CNN_Att(pl.LightningModule):
    def __init__(self, conf) -> None:
        super(CNN_Att, self).__init__()
        self.conf = conf
        self.Spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=self.conf.features.n_fft,
            hop_length=self.conf.features.hop_length,
            win_length=self.conf.features.win_length,
            center=self.conf.features.center,
            pad_mode=self.conf.features.pad_mode,
            power=None,
            return_complex=True
            #freeze_parameters=True
        )
        self.LogMel = nn.Sequential(
            torchaudio.transforms.MelScale(
                sample_rate=self.conf.features.sample_rate,
                n_stft=self.conf.features.n_fft//2+1,
                n_mels=self.conf.features.n_mels,
                f_min=self.conf.features.f_min,
                f_max=self.conf.features.f_max,
            ),
            torchaudio.transforms.AmplitudeToDB()
        )
        self.SpectAugment = SpecAugment(
            time_mask_param=self.conf.augmentation.time_mask,
            freq_mask_param=self.conf.augmentation.freq_mask,
            n_fft=self.conf.features.n_fft,
            rate=1.2
        )
        #self.encoder = nn.Sequential(
        #    ConvBlock2D(1, 128),
        #    ConvBlock2D(128, 256),
        #    ConvBlock2D(256, 512),
        #)
        self.encoder = timm.create_model('resnest50d', pretrained=True, num_classes=0)
        self.fc = nn.Linear(2048, self.conf.training.class_num)

        self.criterion = BCELossModified()

        multiclass = True if self.conf.training.class_num > 2 else False
        self.train_acc = torchmetrics.Accuracy(
        )

        self.train_auroc = torchmetrics.AUROC(
        )

        self.val_acc = torchmetrics.Accuracy(
        )

        self.val_auroc = torchmetrics.AUROC(
        )

        self.tb_writer = SummaryWriter()

    def forward(self, x) -> Dict:
        """
        input : Tensor (batch_size, data_length)
        """
        # create feature
        x = self.Spectrogram(x)
        x = self.SpectAugment(x)
        x = self.LogMel(x)

        # repeat feature for use in ResNest50
        x = torch.unsqueeze(x, 1)
        x = x.repeat(1, 3, 1, 1)
        # Feed into ResNeSt
        x = self.encoder(x)
        # fully connected
        x = self.fc(x)

        return x.squeeze()

    def training_step(self, batch, batch_idx):
        X = batch['audio']
        Y = batch['hot_target'].float()
        Y_out = self(X)
        
        train_loss = self.criterion(Y_out, Y)
        target = batch['target'].int()
        Y_pred = Y_out
        self.train_acc(Y_pred, target)
        #self.train_auroc(Y_pred, target)

        self.log('train_loss', train_loss)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False, prog_bar=True)
        #self.log('train_auroc', self.train_auroc, on_step=True, on_epoch=False, prog_bar=True)
        
        return {'loss': train_loss}

    def validation_step(self, batch, batch_idx):
        X = batch['audio']
        Y = batch['hot_target'].float()
        Y_out = self(X)

        val_loss = self.criterion(Y_out, Y)
        target = batch['target'].int()
        Y_pred = Y_out
        self.val_acc(Y_pred, target)
        #self.val_auroc(Y_pred, target)

        self.log('val_loss', val_loss)
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=False)
        #self.log('val_auroc', self.val_auroc, on_step=True, on_epoch=False, prog_bar=True)

        return {'val_loss': val_loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr = self.conf.training.lr,
            weight_decay=1E-4,
            amsgrad=True
        )

        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.conf.training.factor,
                patience=self.conf.training.patience,
                verbose=True
            ), 'monitor': 'val_loss'}

        return {'optimizer':optimizer, 'lr_scheduler': lr_scheduler}

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
