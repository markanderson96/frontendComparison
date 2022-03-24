import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import torchaudio
from resnest import resnest50

from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, Tuple
from torch import Tensor

from losses import BCELossModified
from efficientnet_pytorch import EfficientNet
from modules import ConvBlock1D, ConvBlock2D, AttnBlock, init_layer
from specaugment import SpecAugment
import leaf_audio_pytorch.frontend as leaf_audio
from frontends.pcen import PCENLayer
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
            _frontend_layers.append(
                PCENLayer(
                   n_mels=self.conf.features.n_mels
                )
            )

        elif self.conf.features.frontend == 'td':
            _frontend_layers.append(
                TDFbanks(
                    mode='learnfbanks',
                    nfilters=self.conf.features.n_mels,
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
        self.efficient_net = EfficientNet.from_pretrained(
            'efficientnet-b0', 
            in_channels=input_channels,
            num_classes=1
        )

        #self.resnest = resnest50(input_channels=input_channels)

        self.criterion = nn.BCEWithLogitsLoss()#BCELossModified()
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
        #x = self.resnest(x)
        x = self.efficient_net(x)
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        # x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
        # x_elements = x.shape[1]*x.shape[2]*x.shape[3]
        # x = x.view(-1, x_elements)
        # x = F.relu(nn.Linear(x_elements, 256, device='cuda:0')(x))
        # x = F.relu(nn.Linear(256, 32, device='cuda:0')(x))
        # x = nn.Linear(32, 1, device='cuda:0')(x)      

        return x.squeeze()

    def training_step(self, batch, batch_idx):
        X = batch[0]
        Y = batch[1].float()
        Y_out = self(X)
        
        train_loss = self.criterion(Y_out, Y)
        target = batch[1].int()
        Y_pred = Y_out
        self.train_acc(Y_pred, target)
        #self.train_auroc(Y_pred, target)

        self.log('train_loss', train_loss)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False, prog_bar=True)
        #self.log('train_auroc', self.train_auroc, on_step=False, on_epoch=True,)
        
        return {'loss': train_loss}

    def validation_step(self, batch, batch_idx):
        X = batch[0]
        Y = batch[1].float()
        Y_out = self(X)

        val_loss = self.criterion(Y_out, Y)
        target = batch[1].int()
        Y_pred = Y_out
        self.val_acc(Y_pred, target)
        #self.val_auroc(Y_pred, target)

        self.log('val_loss', val_loss)
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=False)
        #self.log('val_auroc', self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)

        return {'val_loss': val_loss}

    def test_step(self, batch, batch_idx):
        X = batch[0]
        Y = batch[1].int()
        Y_out = torch.where(self(X) > -0.3, 1, 0)
        acc = torchmetrics.functional.accuracy(Y_out, Y)

        self.log('test_acc', acc, on_step=True)

    def predict_step(self, batch, batch_idx):
        X = batch[0]
        Y = batch[1].float()
        Y_out = torch.where(self(X) > -0.0, 1, 0)
        return Y_out33
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr = self.conf.training.lr,
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

class CNN_Att(pl.LightningModule):
    def __init__(self, conf) -> None:
        super(CNN_Att, self).__init__()
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
            _frontend_layers.append(
                PCENLayer(
                   n_mels=self.conf.features.n_mels
                )
            )

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
        self.encoder = nn.Sequential(
            ConvBlock2D(1, 128),
            ConvBlock2D(128, 256),
            ConvBlock2D(256, 512),
        )
        #self.encoder = timm.create_model('tf_efficientnet_b0_ns', pretrained=True)
        self.fc1 = nn.Linear(512, 1024)
        self.bn1 = nn.BatchNorm2d(self.conf.features.n_mels)
        self.attn_block = AttnBlock(1024, self.conf.training.class_num)
        init_layer(self.fc1)
        init_layer(self.bn1)

        self.criterion = BCELossModified()

        multiclass = True if self.conf.training.class_num > 2 else False
        self.train_acc = torchmetrics.Accuracy(
            num_classes=self.conf.training.class_num,
            multiclass=multiclass
        )
        
        self.val_acc = torchmetrics.Accuracy(
            num_classes=self.conf.training.class_num,
            multiclass=multiclass
        )

        self.tb_writer = SummaryWriter()

    def forward(self, x) -> Dict:
        """
        input : Tensor (batch_size, data_length)
        """
        # create feature
        x = torch.unsqueeze(x, 1)
        x = self.frontend(x)
        # disgusting workaround below
        if self.conf.features.frontend == 'leaf':  
            x = torch.unsqueeze(x, 1)
        #x = x.unsqueeze(dim=1)
        # Extract features
        #x = x.expand(x.shape[0], 3, x.shape[2], x.shape[3]) # Output shape: (batch size, channels=3, time, frequency)
        x = self.encoder(x)
        x = torch.mean(x, dim=3)
        x = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        # dense layer
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5)
        # attention
        clip_out, segment_out = self.attn_block(x)
        segment_out = segment_out.transpose(1, 2)

        output_dict = {
            'clip_out': clip_out,
            'segment_out': segment_out
        }

        return output_dict

    def training_step(self, batch, batch_idx):
        X = batch[0]
        Y = batch[1].float()
        Y_out = self(X)['clip_out'].squeeze()

        train_loss = self.criterion(Y_out, Y)
        target = batch[1].int()
        Y_pred = Y_out
        self.train_acc(Y_pred, target)

        self.log('train_loss', train_loss)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False, prog_bar=True)
        
        return {'loss': train_loss}

    def validation_step(self, batch, batch_idx):
        X = batch[0]
        Y = batch[1].float()
        Y_out = self(X)['clip_out'].squeeze()
        
        val_loss = self.criterion(Y_out, Y)
        target = batch[1].int()
        Y_pred = Y_out
        self.val_acc(Y_pred, target)

        self.log('val_loss', val_loss)
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=False, prog_bar=True)

        return {'val_loss': val_loss}

    def test_step(self, batch, batch_idx):
        X = batch[0]
        Y = batch[1].int()
        Y_out = torch.where(self(X)['clip_out'] > 0.5, 1, 0)
        acc = torchmetrics.functional.accuracy(Y_out, Y)
        f1 = torchmetrics.functional.f1(Y_out, Y)

        self.log('test_f1', f1, on_step=True)
        self.log('test_acc', acc, on_step=True)

    def predict_step(self, batch, batch_idx):
        X = batch[0]
        Y = batch[1].float()
        Y_out = torch.where(self(X)['clip_out'] > 0.5, 1, 0)
        return Y_out

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
