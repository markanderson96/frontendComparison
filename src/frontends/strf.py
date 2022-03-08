#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2019 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr
# Juan Manuel Coria

from typing import Optional
from typing import Text

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from torch.nn.modules.conv import _ConvNd
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
from typing import Optional

from torch.nn.utils.rnn import PackedSequence

def get_info(sequences):
    """Get info about batch of sequences

    Parameters
    ----------
    sequences : `torch.Tensor` or `PackedSequence`
        Batch of sequences given as a `torch.Tensor` of shape
        (batch_size, n_samples, n_features) if sequences all share the same
        length, or as a `PackedSequence` if they do not.

    Returns
    -------
    batch_size : `int`
        Number of sequences in batch.
    n_features : `int`
        Number of features.
    device : `torch.device`
        Device.
    """

    packed_sequences = isinstance(sequences, PackedSequence)

    if packed_sequences:
        _, n_features = sequences.data.size()
        batch_size = sequences.batch_sizes[0].item()
        device = sequences.data.device
    else:
        # check input feature dimension
        batch_size, _, n_features = sequences.size()
        device = sequences.device

    return batch_size, n_features, device

class AmplitudeToDB(torch.jit.ScriptModule):
    # type: (Tensor, float, float, float, Optional[float]) -> Tensor
    r"""Copy pasted pytorch/audio due to version compatibility
    Turn a tensor from the power/amplitude scale to the decibel scale.
    This output depends on the maximum value in the input tensor, and so
    may return different values for an audio clip split into snippets vs. a
    a full clip.
    Args:
        x (torch.Tensor): Input tensor before being converted to decibel scale
        multiplier (float): Use 10. for power and 20. for amplitude
        amin (float): Number to clamp ``x``
        db_multiplier (float): Log10(max(reference value and amin))
        top_db (Optional[float]): Minimum negative cut-off in decibels.
        A reasonable number
            is 80. (Default: ``None``)
    Returns:
        torch.Tensor: Output tensor in decibel scale
    """
    def __init__(self, stype='power', top_db=None):
        super(AmplitudeToDB, self).__init__()
        self.stype = stype
        if top_db is not None and top_db < 0:
            raise ValueError('top_db must be positive value')
        self.top_db = torch.jit.Attribute(top_db, Optional[float])
        self.multiplier = 10.0 if stype == 'power' else 20.0
        self.amin = 1e-10
        self.ref_value = 1.0
        self.db_multiplier = math.log10(max(self.amin, self.ref_value))

    def forward(self, x):
        x_db = self.multiplier * torch.log10(torch.clamp(x, min=self.amin))
        x_db -= self.multiplier * self.db_multiplier

        if self.top_db is not None:
            x_db = x_db.clamp(min=x_db.max().item() - self.top_db)

        return x_db

class STRFNet(nn.Module):
    """STRFNet (learnable) feature extraction
    Parameters
    ----------
    waveform_normalize : `bool`, optional
        Standardize waveforms (to zero mean and unit standard deviation) and
        apply (learnable) affine transform. Defaults to True.
    instance_normalize : `bool`, optional
        Standardize internal representation (to zero mean and unit standard
        deviation) and apply (learnable) affine transform. Defaults to True.
    """
    def __init__(
        self,
        waveform_normalize=True,
        sample_rate=16000,
        num_gabor_filters=64,
        kernel_size=(9, 111),
        stride=[1, 1],
        n_mels=64,
        n_fft=2048,
        window='hamming',
        duration=0.025,
        step=0.01,
        pre_lstm_layer_bool=False,
        window_pre_lstm=1,
        mel_trainable=False,
        stft_trainable=False,
        pre_lstm_compression_dim=64,
        dropout_pre_lstm=0.0,
        gabor_mode='concat',
        instance_normalize=False,
    ):
        super().__init__()

        # check parameters values
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self.step = step
        self.duration = self.n_fft / self.sample_rate
        self.hop_length = int(self.step * self.sample_rate)
        self.num_gabor_filters = num_gabor_filters
        self.kernel_size = kernel_size
        self.waveform_normalize = waveform_normalize
        self.pre_lstm_layer_bool = pre_lstm_layer_bool
        self.window_pre_lstm = window_pre_lstm
        self.stft_trainable = stft_trainable
        self.n_mels = n_mels
        self.stride = stride
        self.instance_normalize = instance_normalize
        # Waveform normalization
        self.waveform_normalize = waveform_normalize
        self.pre_lstm_compression_dim = pre_lstm_compression_dim
        if self.waveform_normalize:
            self.waveform_normalize_ = torch.nn.InstanceNorm1d(1, affine=True)
        # self.n_features_ = n_features
        # self.n_classes_ = n_classes
        self.duration = duration
        self.gabor_mode = gabor_mode
        self.mel_layer = torchaudio.transforms.MelSpectrogram(
            sample_rate =self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            center=False,
        )
        self.lay_ampl = AmplitudeToDB(top_db=80)
        self.gabor_layer = STRFConv2d(
            in_channels=1,
            out_channels=self.num_gabor_filters,
            kernel_size=self.kernel_size,
            padding=(int(self.kernel_size[0] // 2),
                     int(self.kernel_size[1] // 2)),
            stride=self.stride,
            n_features=self.n_mels,
        )
        # self.gabor_activation = nn.ReLU(inplace=True)
        self.gabor_activation = nn.LeakyReLU(negative_slope=0.2)
        self.dropout_pre_lstm_layer = nn.Dropout2d(p=0.2)
        if self.gabor_mode == 'concat':
            input_dim_conv = 2 * self.n_mels * self.num_gabor_filters
        else:
            input_dim_conv = self.n_mels * self.num_gabor_filters
        if self.gabor_mode == 'concat':
            input_dim_conv = 2 * self.n_mels
        else:
            input_dim_conv = self.n_mels
        if self.pre_lstm_layer_bool:
            self.pre_lstm_layer = nn.Conv2d(
                self.num_gabor_filters,
                self.pre_lstm_compression_dim,
                (input_dim_conv, window_pre_lstm),
                stride=(window_pre_lstm, 1),
                padding=int(window_pre_lstm // 2))
        if self.instance_normalize:
            self.instance_norm_layer = nn.InstanceNorm1d(1,
                                                         affine=True)

    def forward(self, waveforms):
        """Extract STRFNet features
        Parameters
        ----------
        waveforms : (batch_size, n_samples, 1)
            Batch of waveforms
        Returns
        -------
        features : (batch_size, n_frames, out_channels[-1])
        """

        if isinstance(waveforms, PackedSequence):
            msg = (f'{self.__class__.__name__} does not support batches '
                   f'containing sequences of variable length.')
            raise ValueError(msg)

        batch_size, n_features, device = get_info(waveforms)
        output = waveforms
        if self.waveform_normalize:
            #output = output.transpose(1, 2)
            output = F.instance_norm(output)
            # output = output.transpose(1, 2)
        output = self.mel_layer(output)
        output = self.lay_ampl(output)
        if self.gabor_mode == 'abs':
            output = torch.pow(self.gabor_layer(
                output, use_real=True), 2) + torch.pow(self.gabor_layer(
                    output, use_real=False), 2)
            output = torch.pow(output, 0.5)
        elif self.gabor_mode == 'real':
            output = self.gabor_layer(output, use_real=True)
        elif self.gabor_mode == 'imag':
            output = self.gabor_layer(output, use_real=False)
        elif self.gabor_mode == 'concat':
            output_real = self.gabor_layer(output, use_real=True)
            output_imag = self.gabor_layer(output, use_real=False)
            output = torch.cat((output_real, output_imag), 1)
        elif self.gabor_mode == 'pass':
            output = output
        output_shape = output.shape
        output = self.dropout_pre_lstm_layer(output)
        if self.gabor_mode == 'concat':
            output = output.reshape(
                output.size(0), self.num_gabor_filters,
                2 * int(self.n_mels / self.stride[0]), output.size(3))
        else:
            output = output.reshape(
                output.size(0), self.num_gabor_filters,
                int(self.n_mels / self.stride[0]), output.size(3))
        if self.pre_lstm_layer_bool:
            # Pre-lstm-layer
            output = self.pre_lstm_layer(output)
            # apply non-linear activation function
            output = F.relu(output, inplace=True)
            output = output.reshape(output.size(0),
                                    output.size(3),
                                    output.size(1))
        return output

    def dimension():
        doc = "Output features dimension."

        def fget(self):
            return self.pre_lstm_compression_dim

        return locals()

    dimension = property(**dimension())
 
class STRFConv2d(_ConvNd):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 padding_mode='zeros',
                 device=None,
                 n_features=64,
                 classic_freq_unit_init=True):

        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super(STRFConv2d,
              self).__init__(in_channels, out_channels,
                             kernel_size, stride, padding, dilation, False,
                             _pair(0), groups, bias, padding_mode)
        self.n_features = n_features

        self.theta = np.random.vonmises(0, 0, (out_channels, in_channels))
        self.gamma = np.random.vonmises(0, 0, (out_channels, in_channels))
        self.psi = np.random.vonmises(0, 0, (out_channels, in_channels))
        self.gamma = nn.Parameter(torch.Tensor(self.gamma))
        self.psi = nn.Parameter(torch.Tensor(self.psi))
        if classic_freq_unit_init:
            self.freq = (np.pi / 2) * 1.41**(
                -np.random.uniform(0, 5, size=(out_channels, in_channels)))
        else:
            self.freq = np.random.rayleigh(1.1,
                                           size=(out_channels, in_channels))
        # betaprime.rvs(1, 5, size=(out_channels, in_channels))

        self.freq = nn.Parameter(torch.Tensor(self.freq))
        self.theta = nn.Parameter(torch.Tensor(self.theta))

        self.sigma_x = 2 * 1.41**(np.random.uniform(
            0, 6, (out_channels, in_channels)))
        self.sigma_x = nn.Parameter(torch.Tensor(self.sigma_x))
        self.sigma_y = 2 * 1.41**(np.random.uniform(
            0, 6, (out_channels, in_channels)))
        self.sigma_y = nn.Parameter(torch.Tensor(self.sigma_y))
        self.f0 = torch.ceil(torch.Tensor([self.kernel_size[0] / 2]))[0]
        self.t0 = torch.ceil(torch.Tensor([self.kernel_size[1] / 2]))[0]

    def forward(self, sequences, use_real=True):
        # batch_size, n_features, device = get_info(sequences)
        packed_sequences = isinstance(sequences, PackedSequence)
        if packed_sequences:
            device = sequences.data.device
        else:
            device = sequences.device
        sequences = sequences.reshape(
            sequences.size(0), 1, self.n_features, -1)
        grid = [
            torch.linspace(-self.f0 + 1, self.f0, self.kernel_size[0]),
            torch.linspace(-self.t0 + 1, self.t0, self.kernel_size[1])
        ]
        f, t = torch.meshgrid(grid)
        f = f.to(device)
        t = t.to(device)
        weight = torch.empty(self.weight.shape, requires_grad=False)
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                sigma_x = self.sigma_x[i, j].expand_as(t)
                sigma_y = self.sigma_y[i, j].expand_as(t)
                # omega_freq = self.omega_freq[i, j].expand_as(y)
                # omega_time = self.omega_time[i, j].expand_as(y)
                freq = self.freq[i, j].expand_as(t)
                theta = self.theta[i, j].expand_as(t)
                gamma = self.gamma[i, j].expand_as(t)
                psi = self.psi[i, j].expand_as(t)
                rotx = t * torch.cos(theta) + f * torch.sin(theta)
                roty = -t * torch.sin(theta) + f * torch.cos(theta)
                rot_gamma = t * torch.cos(gamma) + f * torch.sin(gamma)
                g = torch.zeros(t.shape)
                g = torch.exp(-0.5 * ((f**2) / (sigma_x + 1e-3)**2 +
                                      (t**2) / (sigma_y + 1e-3)**2))
                if use_real:
                    # g = g * torch.cos(2 * np.pi * (omega_freq * x +
                    # omega_time * y) + psi)
                    g = g * torch.cos(freq * rot_gamma)
                else:
                    g = g * torch.sin(freq * rot_gamma)
                g = g / (2 * np.pi * sigma_x * sigma_y)
                weight[i, j] = g
                self.weight.data[i, j] = g
        weight = weight.to(device)
        return F.conv2d(sequences, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
