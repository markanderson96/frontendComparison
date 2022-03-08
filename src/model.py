import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
import torchaudio
from resnest import resnest50

from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, Tuple
from torch import Tensor

from losses import BCELossModified
from specaugment import SpecAugment
import leaf_audio_pytorch.frontend as leaf_audio
from frontends.sincnet import SincNet
from frontends.strf import STRFNet
from frontends.TDFbanks import TDFbanks

class Model(pl.LightningModule):
    def __init__(self, conf) -> None:
        super(Model, self).__init__()
        self.conf = conf
        
        # empty list for frontend layers, 
        # all the layers are already defined but depending on
        # what is specified in the config, 
        # only some of them will be loaded into the model
        input_channels = 1 # annoying workaround for STRF
        _frontend_layers = []
        if self.conf.features.frontend == 'logmel':
            if self.conf.features.spectaugment:
                _frontend_layers.append(
                    torchaudio.transforms.Spectrogram(
                        n_fft=self.conf.features.n_fft,
                        hop_length=self.conf.features.hop_length,
                        win_length=self.conf.features.win_length,
                        center=self.conf.features.center,
                        pad_mode=self.conf.features.pad_mode,
                        power=None,
                        return_complex=True
                    )
                )
                _frontend_layers.append(
                    SpecAugment(
                        time_mask_param=self.conf.augmentation.time_mask,
                        freq_mask_param=self.conf.augmentation.freq_mask,
                        n_fft=self.conf.features.n_fft,
                        rate=1.2
                    )
                )
            else:
                _frontend_layers.append(
                    torchaudio.transforms.Spectrogram(
                        n_fft=self.conf.features.n_fft,
                        hop_length=self.conf.features.hop_length,
                        win_length=self.conf.features.win_length,
                        center=self.conf.features.center,
                        pad_mode=self.conf.features.pad_mode,
                    )
                )
            _frontend_layers.append(
                nn.Sequential(
                    torchaudio.transforms.MelScale(
                        sample_rate=self.conf.features.sample_rate,
                        n_stft=self.conf.features.n_fft//2+1,
                        n_mels=self.conf.features.n_mels,
                        f_min=self.conf.features.f_min,
                        f_max=self.conf.features.f_max,
                    ),
                    torchaudio.transforms.AmplitudeToDB()
                )
            )

        elif self.conf.features.frontend == 'mel':        
            if self.conf.features.spectaugment:
                _frontend_layers.append(
                    torchaudio.transforms.Spectrogram(
                        n_fft=self.conf.features.n_fft,
                        hop_length=self.conf.features.hop_length,
                        win_length=self.conf.features.win_length,
                        center=self.conf.features.center,
                        pad_mode=self.conf.features.pad_mode,
                        power=None,
                        return_complex=True
                    )
                )
                _frontend_layers.append(
                    SpecAugment(
                        time_mask_param=self.conf.augmentation.time_mask,
                        freq_mask_param=self.conf.augmentation.freq_mask,
                        n_fft=self.conf.features.n_fft,
                        rate=1.2
                    )
                )
            else:
                _frontend_layers.append(
                    torchaudio.transforms.Spectrogram(
                        n_fft=self.conf.features.n_fft,
                        hop_length=self.conf.features.hop_length,
                        win_length=self.conf.features.win_length,
                        center=self.conf.features.center,
                        pad_mode=self.conf.features.pad_mode,
                    )
                )
            _frontend_layers.append(
                torchaudio.transforms.MelScale(
                    sample_rate=self.conf.features.sample_rate,
                    n_stft=self.conf.features.n_fft//2+1,
                    n_mels=self.conf.features.n_mels,
                    f_min=self.conf.features.f_min,
                    f_max=self.conf.features.f_max,
                )
            )

        elif self.conf.features.frontend == 'spect':
            if self.conf.features.spectaugment:
                _frontend_layers.append(
                    torchaudio.transforms.Spectrogram(
                        n_fft=self.conf.features.n_fft,
                        hop_length=self.conf.features.hop_length,
                        win_length=self.conf.features.win_length,
                        center=self.conf.features.center,
                        pad_mode=self.conf.features.pad_mode,
                        power=None,
                        return_complex=True
                    )
                )
                _frontend_layers.append(
                    SpecAugment(
                        time_mask_param=self.conf.augmentation.time_mask,
                        freq_mask_param=self.conf.augmentation.freq_mask,
                        n_fft=self.conf.features.n_fft,
                        rate=1.2
                    )
                )
            else:
                _frontend_layers.append(
                    torchaudio.transforms.Spectrogram(
                        n_fft=self.conf.features.n_fft,
                        hop_length=self.conf.features.hop_length,
                        win_length=self.conf.features.win_length,
                        center=self.conf.features.center,
                        pad_mode=self.conf.features.pad_mode,
                    )
                )

        elif self.conf.features.frontend == 'pcen':
            if self.conf.features.spectaugment:
                _frontend_layers.append(
                    torchaudio.transforms.Spectrogram(
                        n_fft=self.conf.features.n_fft,
                        hop_length=self.conf.features.hop_length,
                        win_length=self.conf.features.win_length,
                        center=self.conf.features.center,
                        pad_mode=self.conf.features.pad_mode,
                        power=None,
                        return_complex=True
                    )
                )
                _frontend_layers.append(
                    SpecAugment(
                        time_mask_param=self.conf.augmentation.time_mask,
                        freq_mask_param=self.conf.augmentation.freq_mask,
                        n_fft=self.conf.features.n_fft,
                        rate=1.2
                    )
                )
            else:
                _frontend_layers.append(
                    torchaudio.transforms.Spectrogram(
                        n_fft=self.conf.features.n_fft,
                        hop_length=self.conf.features.hop_length,
                        win_length=self.conf.features.win_length,
                        center=self.conf.features.center,
                        pad_mode=self.conf.features.pad_mode,
                    )
                )
            _frontend_layers.append(
                torchaudio.transforms.MelScale(
                    sample_rate=self.conf.features.sample_rate,
                    n_stft=self.conf.features.n_fft//2+1,
                    n_mels=self.conf.features.n_mels,
                    f_min=self.conf.features.f_min,
                    f_max=self.conf.features.f_max,
                )
            )
            _frontend_layers.append(self.pcen)

        elif self.conf.features.frontend == 'td':
            _frontend_layers.append(
                TDFbanks(
                    mode='learnfbanks',
                    nfilters=80,
                    samplerate=self.conf.features.sample_rate
                ) 
            )

        # DO NOT USE
        # SincNet outputs a 1-d representation needs
        # its own network simply comprising of fc
        # of arch: Dense (input) -> Dense (hidden) -> Dense (output)
        elif self.conf.features.frontend == 'sinc':
            _frontend_layers.append(
                SincNet() 
            )

        elif self.conf.features.frontend == 'leaf':
            _frontend_layers.append(
                leaf_audio.Leaf(
                    sample_rate=self.conf.features.sample_rate,
                )
            )

        elif self.conf.features.frontend == 'strf':
            _frontend_layers.append(
                STRFNet(
                    sample_rate=self.conf.features.sample_rate,
                    n_mels=self.conf.features.n_mels,
                    n_fft=self.conf.features.n_fft
                )
            )
            input_channels=64

        else:
            raise Exception("Must specify a valid front-end in config/config.yaml")

        self.frontend = nn.Sequential(*_frontend_layers)        
        self.resnest = resnest50(input_channels=input_channels)
        #self.fc = nn.Linear(2048, self.conf.training.class_num)

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

    def forward(self, x) -> Tensor:
        """
        input : Tensor (batch_size, channels, data_length)
        """
        # create feature
        x = torch.unsqueeze(x, 1)
        x = self.frontend(x)
        # disgusting workaround below
        if self.conf.features.frontend == 'leaf':  
            x = torch.unsqueeze(x, 1)
        # Feed into ResNeSt
        x = self.resnest(x)

        return x.squeeze()

    def training_step(self, batch, batch_idx):
        X = batch['audio']
        Y = batch['target'].float()
        Y_out = self(X)
        
        train_loss = self.criterion(Y_out, Y)
        target = batch['target'].int()
        Y_pred = Y_out
        self.train_acc(Y_pred, target)
        self.train_auroc(Y_pred, target)

        self.log('train_loss', train_loss)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False, prog_bar=True)
        #self.log('train_auroc', self.train_auroc, on_step=False, on_epoch=True,)
        
        return {'loss': train_loss}

    def validation_step(self, batch, batch_idx):
        X = batch['audio']
        Y = batch['target'].float()
        Y_out = self(X)

        val_loss = self.criterion(Y_out, Y)
        target = batch['target'].int()
        Y_pred = Y_out
        self.val_acc(Y_pred, target)
        self.val_auroc(Y_pred, target)

        self.log('val_loss', val_loss)
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=False)
        self.log('val_auroc', self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)

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
