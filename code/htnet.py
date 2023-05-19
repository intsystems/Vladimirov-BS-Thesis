import torch
import torch.nn as nn
import torch.nn.functional as F

# Load utility functions for custom HTNet layers
from utils import apply_hilbert_tf, proj_to_roi


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

        self.block1 = nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength // 2), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)

        if projectROIs:
            self.depthwise1 = nn.Conv2d(F1, F1 * D, (ROIs, 1), groups=F1, bias=False)
        else:
            self.depthwise1 = nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False)

        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.activation1 = nn.ELU()
        self.pooling1 = nn.AvgPool2d((1, 4))
        self.dropout1 = dropoutType(p=dropoutRate)

        self.block2 = nn.Conv2d(F1 * D, F2, (1, kernLength_sep), padding=(0, kernLength_sep // 2), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.activation2 = nn.ELU()
        self.pooling2 = nn.AvgPool2d((1, 8))
        self.dropout2 = dropoutType(p=dropoutRate)

        self.flatten = nn.Flatten()
        self.dense = nn.Linear(F2 * (Samples // (4 * 8)), nb_classes)
        self.softmax = nn.Softmax(dim=1)

    def max_norm_adjusting(self, weights, max_norm):
        with torch.no_grad():
            norm = weights.norm(2, dim=0, keepdim=True).clamp(min=max_norm / 2)
            desired = torch.clamp(norm, max=max_norm)
            weights *= (desired / norm)

    def forward(self, x, roi=None):
        # x.shape = (1, Chans, Samples)
        # roi.shape = (1, ROIs, Chans)
        x = self.block1(x)
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

        self.max_norm_adjusting(self.depthwise1.weight, 1.0)
        x = self.depthwise1(x)
        x = self.batchnorm2(x)
        x = self.activation1(x)
        x = self.pooling1(x)
        x = self.dropout1(x)
        x = self.block2(x)
        x = self.batchnorm3(x)
        x = self.activation2(x)
        x = self.pooling2(x)
        x = self.dropout2(x)
        x = self.flatten(x)

        self.max_norm_adjusting(self.dense.weight, self.max_dense_layer_norm)
        x = self.dense(x)
        x = self.softmax(x)
        return x
