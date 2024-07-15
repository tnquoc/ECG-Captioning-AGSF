import torch.nn as nn


class SpectrogramMiniResnet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv2D_1 = nn.Conv2d(in_channels, 16, kernel_size=(3, 3), stride=1, padding=(2, 0))
        self.conv2D_2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=(2, 0))
        self.conv2D_3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 1), padding=(2, 0))
        self.bn_1 = nn.BatchNorm2d(64)
        self.relu_1 = nn.ReLU()
        self.pool_1 = nn.AvgPool2d((2, 2))
        self.conv2D_4 = nn.Conv2d(64, 128, kernel_size=(3, 2), stride=1, padding=(2, 0))
        self.conv2D_5 = nn.Conv2d(128, 256, kernel_size=(3, 2), stride=(2, 1), padding=(2, 0))
        self.bn_2 = nn.BatchNorm2d(256)
        self.relu_2 = nn.ReLU()
        self.dropout_1 = nn.Dropout(0.3)
        self.conv2D_6 = nn.Conv2d(256, 512, kernel_size=(2, 2), stride=1, padding=(1, 0))
        self.pool_2 = nn.AvgPool2d((2, 2))

    def forward(self, x):
        x = self.conv2D_1(x)
        x = self.conv2D_2(x)
        x = self.conv2D_3(x)
        x = self.bn_1(x)
        x = self.relu_1(x)
        x = self.pool_1(x)
        x = self.conv2D_4(x)
        x = self.conv2D_5(x)
        x = self.bn_2(x)
        x = self.relu_2(x)
        x = self.dropout_1(x)
        x = self.conv2D_6(x)
        x = self.pool_2(x)

        return x
