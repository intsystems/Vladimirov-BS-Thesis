import warnings
import numpy as np

import torch
import torchcde
import torch.nn as nn
import torch.nn.functional as F

# Load utility functions for custom HTNet layers
from utils import apply_hilbert_tf, proj_to_roi, gen_dropout_mask
from s4 import S4Block


class HTNet(nn.Module):
    def __init__(self, nb_classes, Chans=64, Samples=128,
                 dropoutRate=0.5, kernLength=64, F1=8,
                 D=2, F2=16, dropoutType='Dropout',
                 ROIs=100, useHilbert=False, projectROIs=False, kernLength_sep=16,
                 do_log=False, compute_val='power', data_srate=500, base_split=4):
        super(HTNet, self).__init__()

        if dropoutType == 'SpatialDropout2D':
            dropoutType = nn.Dropout2d
        elif dropoutType == 'Dropout':
            dropoutType = nn.Dropout
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D '
                             'or Dropout, passed as a string.')

        self.useHilbert = useHilbert
        self.projectROIs = projectROIs
        self.compute_val = compute_val
        self.do_log = do_log
        self.data_srate = data_srate
        self.base_split = base_split

        self.temporal_conv = nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength // 2), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)

        if projectROIs:
            self.depthwise_conv = nn.Conv2d(F1, F1 * D, (ROIs, 1), groups=F1, bias=False)
        else:
            self.depthwise_conv = nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False)
        self.depthwise_conv = nn.utils.weight_norm(self.depthwise_conv, name='weight', dim=0)

        self.sequential1 = nn.Sequential(
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            dropoutType(p=dropoutRate)
        )

        depth_conv = nn.Conv2d(F1 * D, F1 * D, (1, kernLength_sep), groups=F1 * D, padding=(0, kernLength_sep // 2))
        point_conv = nn.Conv2d(F1 * D, F2, 1)
        self.separable_conv = nn.Sequential(depth_conv, point_conv)

        self.sequential2 = nn.Sequential(
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            dropoutType(p=dropoutRate)
        )

        self.flatten = nn.Flatten()

        self.dense = nn.Linear(F2 * (Samples // (4 * 8)), nb_classes)
        self.dense = nn.utils.weight_norm(self.dense, name='weight', dim=0)

    def forward(self, x, roi=None):
        # x.shape = (batch_size, 1, Chans, Samples)
        # roi.shape = (batch_size, 1, ROIs, Chans)
        x = self.temporal_conv(x)
        if self.useHilbert:
            if self.compute_val == 'relative_power':
                x1 = apply_hilbert_tf(x, do_log=True, compute_val='power')
                x2 = F.avg_pool2d(x1, (1, x1.size(-1) // self.base_split))
                x2 = x2[..., :1].repeat(1, 1, 1, x1.size(-1))
                x = x1 - x2
            else:
                x = apply_hilbert_tf(x, do_log=self.do_log, compute_val=self.compute_val, data_srate=self.data_srate)

        if self.projectROIs:
            x = proj_to_roi([x, roi])

        x = self.batchnorm1(x)

        x = self.depthwise_conv(x)
        x = self.sequential1(x)
        x = self.separable_conv(x)
        x = self.sequential2(x)
        x = self.flatten(x)

        x = self.dense(x)
        return x


class FastRNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, proj_size=0, dropout=0.0, num_layers=1):
        super(FastRNNLayer, self).__init__()
        self.module = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        if proj_size > 0:
            self.dense = nn.Linear(hidden_size, proj_size)
        else:
            self.dense = nn.Identity()
            #self.dense = nn.utils.weight_norm(self.dense, name='weight')

        self.dropout = 1-dropout
        self.layer_names = np.ravel([[f'weight_hh_l{i}', f'weight_ih_l{i}'] for i in range(num_layers)])
        for layer in self.layer_names:
            # Makes a copy of the weights of the selected layers.
            w = getattr(self.module, layer)
            self.register_parameter(f'{layer}_raw', nn.Parameter(w.data))

    def _setweights(self):
        "Apply dropout to the raw weights."
        input_mask, hidden_mask = gen_dropout_mask(
            self.module.input_size,
            self.module.hidden_size,
            self.training,
            self.dropout,
            self.module.weight_hh_l0.data
        )
        for layer, mask in zip(self.layer_names, (hidden_mask, input_mask)):
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer].data = raw_w * mask

    def forward(self, inp):
        # inp.shape = (batch_size, seq_len, in_dim)
        # outp.shape = (batch_size, seq_len, out_dim)
        with warnings.catch_warnings():
            # To avoid the warning that comes because the weights aren't flattened.
            warnings.filterwarnings("ignore")

            ### set new weights of self.module and call its forward
            self._setweights()
            outp, (h0, c0) = self.module.forward(inp)
            return self.dense(outp), (h0, c0)

    def reset(self):
        if hasattr(self.module, 'reset'): self.module.reset()


class NeuralCDEFunc(nn.Module):
    def __init__(self, hid_size, in_channels):
        super().__init__()
        self.hid_size = hid_size
        self.in_channels = in_channels
        self.linear = nn.Sequential(
            nn.Linear(hid_size, 4, bias=False),
            nn.Linear(4, hid_size * in_channels, bias=False)
        )
        self.activ_func = nn.Tanh()

    def forward(self, t, z):
        z = self.linear(z).view(-1, self.hid_size, self.in_channels)
        return self.activ_func(z)


class NeuralCDE(nn.Module):
    def __init__(self, hidden_size, seq_len, in_channels, out_channels, time=2, dropoutRate=0.5):
        super().__init__()
        self.time = time
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.in_channels = in_channels + 1

        self.func = NeuralCDEFunc(hidden_size, self.in_channels)
        self.dropout = nn.Dropout(p=dropoutRate/5)

        if out_channels > 0:
            self.dense = nn.Linear(hidden_size, out_channels)
        else:
            self.dense = nn.Identity()

    def forward(self, inp):
        # inp.shape = (batch_size, seq_len, in_channels)
        # outp.shape = (batch_size, seq_len, hidden_size/out_channels)
        assert inp.shape[1] == self.seq_len
        assert inp.shape[2] + 1 == self.in_channels
        batch_size, seq_len, in_channels = inp.shape

        t = torch.linspace(0, self.time, seq_len, device=inp.device)
        t_ = t.unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, 1)
        inp = torch.cat([t_, inp], dim=2)  # include time as a channel

        coeffs = torchcde.linear_interpolation_coeffs(inp, t)
        X = torchcde.LinearInterpolation(coeffs, t)
        z0 = torch.rand(batch_size, self.hidden_size, device=inp.device)
        adjoint_params = tuple(self.func.parameters()) + (coeffs,)

        outp = torchcde.cdeint(X=X, func=self.func, z0=z0, t=X._t, method='rk4', adjoint=False)
        return self.dropout(self.dense(outp)), None


class AdaptiveConcatPoolRNN(nn.Module):
    def forward(self, x):
        # input shape bs, ch, ts
        t1 = nn.AdaptiveAvgPool1d(1)(x)
        t2 = nn.AdaptiveMaxPool1d(1)(x)
        t3 = x[:, :, -1]
        out = torch.cat([t1.squeeze(-1), t2.squeeze(-1), t3], 1)  # output shape bs, 3*ch
        return out


class SeqRNN(nn.Module):
    def __init__(self, nb_classes, k_signals=0, dropoutRate=0.5, hid_size=64):
        super().__init__()
        self.seq_model = FastRNNLayer(k_signals, hid_size, num_layers=1, dropout=dropoutRate)
        self.pooling = AdaptiveConcatPoolRNN()
        self.activ = nn.ELU()
        self.dense = nn.Linear(3*hid_size, nb_classes)

    def forward(self, inp, roi):
        # inp.shape = (batch_size, 1, in_dim, seq_len)
        batch_size, _, in_channels, seq_len = inp.shape
        inp = apply_hilbert_tf(inp, do_log=False, compute_val='power', data_srate=500)
        # inp.shape = (batch_size, seq_len, 1, in_dim)
        inp = torch.transpose(torch.transpose(inp, 2, 3), 1, 2)
        inp = inp.reshape(batch_size, seq_len, -1)
        # outp.shape = (batch_size, seq_len, out_dim) ==> (batch_size, out_dim, seq_len) ==> (batch_size, 3*out_dim)
        outp, _ = self.seq_model(inp)
        outp = torch.transpose(outp, 1, 2)
        outp = self.pooling(outp)
        return self.dense(self.activ(outp))


class SeqS4(SeqRNN):
    def __init__(self, nb_classes, k_signals=0, dropoutRate=0.5, hid_size=64):
        super().__init__(nb_classes, k_signals, dropoutRate, hid_size)
        self.seq_model = S4Block(
            d_model=k_signals,
            final_act=None,
            transposed=False,
            weight_norm=True,
            tie_dropout=True,
            dropout=dropoutRate
        )
        self.dense = nn.Linear(3 * k_signals, nb_classes)


class SeqNCDE(SeqRNN):
    def __init__(self, nb_classes, k_signals=0, dropoutRate=0.5, hid_size=64, seq_len=1):
        super().__init__(nb_classes, k_signals, dropoutRate, hid_size)
        self.seq_model = NeuralCDE(hid_size, seq_len, k_signals, -1, dropoutRate=dropoutRate)
