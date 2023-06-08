import warnings

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
                 D=2, F2=16, norm_rate=0.25, dropoutType='Dropout',
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
        self.max_dense_layer_norm = norm_rate

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
        self.softmax = nn.Softmax(dim=1)

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
    def __init__(self, input_size, hidden_size, proj_size=0, dropout=0.0):
        super(FastRNNLayer, self).__init__()
        self.module = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dense = nn.Linear(hidden_size, proj_size)
        self.dense = nn.utils.weight_norm(self.dense, name='weight')
        self.dropout = 1-dropout
        self.layer_names = ['weight_hh_l0', 'weight_ih_l0']
        for layer in self.layer_names:
            # Makes a copy of the weights of the selected layers.
            w = getattr(self.module, layer)
            self.register_parameter(f'{layer}_raw', nn.Parameter(w.data))

    def _setweights(self):
        "Apply dropout to the raw weights."
        ### generate input_mask and hidden_mask (use function gen_dropout_mask)
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


class RNNTemporal(nn.Module):
    def __init__(self, in_channels, input_size, hidden_size, proj_size=0, dropout=0.0):
        super().__init__()
        self.rnn = FastRNNLayer(in_channels*input_size, hidden_size, in_channels*proj_size, dropout)

    def forward(self, inp):
        # inp.shape = (batch_size, in_channels, in_dim, seq_len)
        # input_dim = in_channels * in_dim, output_dim = in_channels * proj_size
        # RNN: (batch_size, seq_len, input_dim) ==> (batch_size, seq_len, output_dim)
        # outp.shape = (batch_size, in_channels, out_dim, seq_len)
        batch_size, in_channels, _, seq_len = inp.shape
        # inp.shape = (batch_size, seq_len, in_channels, in_dim)
        inp = torch.transpose(torch.transpose(inp, 2, 3), 1, 2)
        inp = inp.reshape(batch_size, seq_len, -1)
        outp, _ = self.rnn(inp)
        outp = torch.transpose(outp, 1, 2).reshape(batch_size, in_channels, -1, seq_len)
        return outp


class HTNetWithRNN(HTNet):
    def __init__(self, nb_classes, Chans=64, Samples=128,
                 dropoutRate=0.5, kernLength=64, F1=8,
                 D=2, F2=16, norm_rate=0.25, dropoutType='Dropout',
                 ROIs=100, useHilbert=False, projectROIs=False, kernLength_sep=16,
                 do_log=False, compute_val='power', data_srate=500, base_split=4, hid_size=32, k_signals=0):
        super().__init__(nb_classes, Chans, Samples, dropoutRate, kernLength, F1, D, F2,
                         norm_rate, dropoutType, ROIs, useHilbert, projectROIs, kernLength_sep,
                         do_log, compute_val, data_srate, base_split)
        self.temporal_conv = nn.Sequential(
            RNNTemporal(1, k_signals, hid_size, k_signals, dropoutRate),
            nn.Conv2d(1, F1, 1)
        )
        self.separable_conv = nn.Sequential(
            RNNTemporal(F1 * D, 1, hid_size, 1, dropoutRate),
            nn.Conv2d(F1 * D, F2, 1)
        )


class NeuralCDEFunc(nn.Module):
    def __init__(self, hid_size, in_channels):
        super().__init__()
        self.hid_size = hid_size
        self.in_channels = in_channels
        self.linear = nn.Linear(hid_size, hid_size * in_channels)
        self.activ_func = nn.Tanh()

    def forward(self, t, z):
        z = self.linear(z).view(-1, self.hid_size, self.in_channels)
        return self.activ_func(z)


class NeuralCDE(nn.Module):
    def __init__(self, hidden_size, seq_len, in_channels, out_channels, time=2):
        super().__init__()
        self.time = time
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.in_channels = in_channels + 1

        self.func = NeuralCDEFunc(hidden_size, self.in_channels)
        self.dense = nn.Linear(hidden_size, out_channels)
        self.dense = nn.utils.weight_norm(self.dense, name='weight')

    def forward(self, inp):
        # inp.shape = (batch_size, seq_len, in_channels)
        # outp.shape = (batch_size, seq_len, out_channels)
        assert inp.shape[1] == self.seq_len
        assert inp.shape[2] + 1 == self.in_channels
        batch_size, seq_len, in_channels = inp.shape

        t = torch.linspace(0, self.time, seq_len, device=inp.device)
        t_ = t.unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, 1)
        inp = torch.cat([t_, inp], dim=2)  # include time as a channel

        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(inp, t)
        X = torchcde.CubicSpline(coeffs, t)
        z0 = torch.rand(batch_size, self.hidden_size, device=inp.device)
        adjoint_params = tuple(self.func.parameters()) + (coeffs,)

        outp = torchcde.cdeint(X=X, func=self.func, z0=z0, t=X._t, method='rk4', adjoint=False)
        return self.dense(outp)


class NCDETemporal(nn.Module):
    def __init__(self, in_channels, seq_len, input_size, hidden_size, proj_size=0):
        super().__init__()
        self.ncde = NeuralCDE(hidden_size, seq_len, in_channels*input_size, in_channels*proj_size)

    def forward(self, inp):
        # inp.shape = (batch_size, in_channels, in_dim, seq_len)
        # input_dim = in_channels * in_dim, output_dim = in_channels * proj_size
        # RNN: (batch_size, seq_len, input_dim) ==> (batch_size, seq_len, output_dim)
        # outp.shape = (batch_size, in_channels, out_dim, seq_len)
        batch_size, in_channels, _, seq_len = inp.shape
        # inp.shape = (batch_size, seq_len, in_channels, in_dim)
        inp = torch.transpose(torch.transpose(inp, 2, 3), 1, 2)
        inp = inp.reshape(batch_size, seq_len, -1)
        outp = self.ncde(inp)
        return torch.transpose(outp, 1, 2).reshape(batch_size, in_channels, -1, seq_len)


class HTNetWithNCDE(HTNet):
    def __init__(self, nb_classes, Chans=64, Samples=128,
                 dropoutRate=0.5, kernLength=64, F1=8,
                 D=2, F2=16, norm_rate=0.25, dropoutType='Dropout',
                 ROIs=100, useHilbert=False, projectROIs=False, kernLength_sep=16,
                 do_log=False, compute_val='power', data_srate=500, base_split=4, hid_size=32, k_signals=0):
        super().__init__(nb_classes, Chans, Samples, dropoutRate, kernLength, F1, D, F2,
                         norm_rate, dropoutType, ROIs, useHilbert, projectROIs, kernLength_sep,
                         do_log, compute_val, data_srate, base_split)

        self.temporal_conv = nn.Sequential(
            NCDETemporal(1, Samples, k_signals, hid_size, k_signals),
            nn.Conv2d(1, F1, 1)
        )
        self.separable_conv = nn.Sequential(
            NCDETemporal(F1 * D, Samples // 4, 1, hid_size, 1),
            nn.Conv2d(F1 * D, F2, 1)
        )


class S4Temporal(nn.Module):
    def __init__(self, in_channels, input_size, dropout):
        super().__init__()
        self.s4_model = S4Block(
            d_model=input_size*in_channels,
            final_act='id',
            transposed=False,
            weight_norm=True,
            dropout=dropout
        )

    def forward(self, inp):
        # inp.shape = (batch_size, in_channels, in_dim, seq_len)
        # input_dim = in_channels * in_dim, output_dim = in_channels * proj_size
        # RNN: (batch_size, seq_len, input_dim) ==> (batch_size, seq_len, output_dim)
        # outp.shape = (batch_size, in_channels, out_dim, seq_len)
        batch_size, in_channels, _, seq_len = inp.shape
        # inp.shape = (batch_size, seq_len, in_channels, in_dim)
        inp = torch.transpose(torch.transpose(inp, 2, 3), 1, 2)
        inp = inp.reshape(batch_size, seq_len, -1)
        outp, _ = self.s4_model(inp)
        return torch.transpose(outp, 1, 2).reshape(batch_size, in_channels, -1, seq_len)


class HTNetWithS4(HTNet):
    def __init__(self, nb_classes, Chans=64, Samples=128,
                 dropoutRate=0.5, kernLength=64, F1=8,
                 D=2, F2=16, norm_rate=0.25, dropoutType='Dropout',
                 ROIs=100, useHilbert=False, projectROIs=False, kernLength_sep=16,
                 do_log=False, compute_val='power', data_srate=500, base_split=4, k_signals=1):
        super().__init__(nb_classes, Chans, Samples, dropoutRate, kernLength, F1, D, F2,
                         norm_rate, dropoutType, ROIs, useHilbert, projectROIs, kernLength_sep,
                         do_log, compute_val, data_srate, base_split)

        self.temporal_conv = nn.Sequential(
            S4Temporal(1, k_signals, dropoutRate),
            nn.Conv2d(1, F1, 1)
        )
        self.separable_conv = nn.Sequential(
            S4Temporal(F1 * D, 1, dropoutRate),
            nn.Conv2d(F1 * D, F2, 1)
        )
