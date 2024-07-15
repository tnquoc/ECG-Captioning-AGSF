# ECG ResNet used in the Triage paper

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class BasicBlock(nn.Module):
    """
    This class implements a residual block.
    """
    def __init__(self, in_channels, out_channels, stride, dropout, dilation):
        """
        Initializes BasicBlock object. 

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            stride (int): stride of the convolution.
            dropout (float): probability of an argument to get zeroed in the
                dropout layer.
            dilation (int): amount of dilation in the dilated branch.
        """
        super(BasicBlock, self).__init__()
        kernel_size = 5
        num_branches = 2

        self.branch0 = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, in_channels // num_branches, kernel_size=1,
                      padding=0, stride=1, bias=False),

            nn.BatchNorm1d(in_channels // num_branches),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels // num_branches, out_channels // num_branches,
                      kernel_size=kernel_size,
                      padding=(kernel_size - 1) // 2,
                      stride=stride, bias=False)
        )

        self.branch1 = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, in_channels // num_branches, kernel_size=1,
                      padding=0, stride=1, dilation=1, bias=False),

            nn.BatchNorm1d(in_channels // num_branches),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels // num_branches, out_channels // num_branches,
                      kernel_size=kernel_size,
                      padding=((kernel_size - 1) * dilation) // 2,
                      stride=stride, dilation=dilation, bias=False)
        )

        if in_channels == out_channels and stride == 1:
            self.shortcut = lambda x: x
        else:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, 
                                      padding=0, stride=stride, bias=False)

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is
        transformed through several layer transformations.

        Args:
          x (tensor): input to the block with size NxCxL
        Returns:
          out: outputs of the block
        """
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        r = self.shortcut(x)
        return out.add_(r)


class ECGResNet(nn.Module):
    """
    This class implements the ECG-ResNet in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ResNet object can perform forward.
    """
    def __init__(self, in_length, in_channels, n_grps, N, num_classes, dropout,
                 first_width, stride, dilation):
        """
        Initializes ECGResNet object.

        Args:
            in_length (int): the length of the ECG signal input.
            in_channels (int): number of channels of input (= leads).
            n_grps (int): number of ResNet groups.
            N (int): number of blocks per groups.
            num_classes (int): number of classes of the classification problem.
            stride (tuple): tuple with stride value per block per group.
            dropout (float): the dropout probability.
            first_width (int): the output width of the stem.
            dilation (int): the space between the dilated convolutions.
        """
        super().__init__()
        num_branches = 2
        first_width = first_width * num_branches
        stem = [nn.Conv1d(in_channels, first_width // 2, kernel_size=7, 
                          padding=3, stride=2, dilation=1, bias=False),
                nn.BatchNorm1d(first_width // 2), nn.ReLU(),
                nn.Conv1d(first_width // 2, first_width, kernel_size=1, 
                          padding=0, stride=1, bias=False),
                nn.BatchNorm1d(first_width), nn.ReLU(), nn.Dropout(dropout),
                nn.Conv1d(first_width, first_width, kernel_size=5, 
                          padding=2, stride=1, bias=False)]

        layers = []

        # Double feature depth at each group, after the first
        widths = [first_width]
        for grp in range(n_grps):
            widths.append((first_width) * 2**grp)
        for grp in range(n_grps):
            layers += self._make_group(N, widths[grp], widths[grp + 1],
                                       stride, dropout, dilation)

        layers += [nn.BatchNorm1d(widths[-1]), nn.ReLU(inplace=True)]

        if in_length == 600:
            fclayers1 = [nn.Linear(2432, 256), nn.ReLU(inplace=True),
                         nn.Dropout(dropout), nn.Linear(256, num_classes)]
            fclayers2 = [nn.Linear(1024, 256), nn.ReLU(inplace=True),
                         nn.Dropout(dropout), nn.Linear(256, num_classes)]
        else:
            fclayers1 = [nn.Linear(20096, 256), nn.ReLU(inplace=True),
                         nn.Dropout(dropout), nn.Linear(256, num_classes)]
            fclayers2 = [nn.Linear(5120, 256), nn.ReLU(inplace=True),
                         nn.Dropout(dropout), nn.Linear(256, num_classes)]

        self.stem = nn.Sequential(*stem)
        aux_point = (len(layers) - 2) // 2
        self.features1 = nn.Sequential(*layers[:aux_point])
        self.features2 = nn.Sequential(*layers[aux_point:])
        self.flatten = Flatten()
        self.fc1 = nn.Sequential(*fclayers1)
        self.fc2 = nn.Sequential(*fclayers2)

    def _make_group(self, N, in_channels, out_channels, stride, dropout,
                    dilation):
        """
        Builds a group of blocks.

        Args:
            in_channels (int): number of channels of input
            out_channels (int): number of channels of output
            stride (tuple): tuple of strides of convolutions with length of N
            N (int): number of blocks per groups
            num_classes (int): number of classes of the classification problem
            dropout (float): the dropout probability.
        """
        group = list()
        for i in range(N):
            blk = BasicBlock(in_channels=(in_channels if i == 0 else out_channels),
                             out_channels=out_channels, stride=stride[i],
                             dropout=dropout, dilation=dilation)
            group.append(blk)
        return group

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is
        transformed through several layer transformations.

        Args:
            x (tensor): input to the block with size NxCxL
        Returns:
            out (tuple): outputs of forward pass, first the auxilliary halfway,
                then the final prediction
        """
        # print(x.shape)
        x = self.stem(x)
        x1 = self.features1(x)
        x1out = self.flatten(x1)
        x2 = self.features2(x1)
        x2out = self.flatten(x2)
        return self.fc1(x1out), self.fc2(x2out)
