import torch
from torch import nn


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, pad, str):
        super(Conv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, stride=str, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv1(input)

class Net(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Net, self).__init__()
        self.conv1 = Conv(in_ch, 16, 21, 10, 1)            # input 512; output  512
        self.conv2 = Conv(16, 16, 21, 10, 1)               # input 512; output  512
        self.down1 = nn.MaxPool1d(2, 2)                    # input 512; output  256

        self.conv3 = Conv(16, 32, 13, 6, 1)
        self.conv4 = Conv(32, 32, 13, 6, 1)
        self.down2 = nn.MaxPool1d(2, 2)                    # input 256; output 128


        self.conv5 = Conv(32, 64, 9, 4, 1)
        self.conv6 = Conv(64, 64, 9, 4, 1)
        self.down3 = nn.MaxPool1d(2, 2)                    # input 128; output 64

        self.conv7 = Conv(64, 128, 7, 3, 1)
        self.conv8 = Conv(128, 128, 7, 3, 1)
        self.down4 = nn.MaxPool1d(2, 2)                    # input 64; output  32

        self.conv9 = nn.ConvTranspose1d(128, 64, 2, 2)     # input 32; output 64
        self.conv10 = Conv(64, 64, 5, 2, 1)
        self.conv11 = Conv(64, 64, 5, 2, 1)

        self.conv12 = nn.ConvTranspose1d(64, 32, 2, 2)     # input 64; output 128
        self.conv13 = Conv(32, 32, 13, 6, 1)
        self.conv14 = Conv(32, 32, 13, 6, 1)

        self.conv15 = nn.ConvTranspose1d(32, 16, 2, 2)     # input 128; output 256
        self.conv16 = Conv(16, 16, 15, 7, 1)
        self.conv17 = Conv(16, 16, 15, 7, 1)

        self.conv18 = nn.ConvTranspose1d(16, 2, 2, 2)      # input 256; output 512
        self.conv19 = Conv(2, 2, 7, 3, 1)


        self.classifier = nn.Sequential(
            nn.Linear(2*512, 2048),
            nn.ReLU(True),
            nn.Dropout(p=0.3),
            nn.Linear(2048, out_ch)
        )

    def forward(self, x):

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.down1(out)

        out = self.conv3(out)
        out = self.conv4(out)
        out = self.down2(out)

        out = self.conv5(out)
        out = self.conv6(out)
        out = self.down3(out)

        out = self.conv7(out)
        out = self.conv8(out)
        out = self.down4(out)

        out = self.conv9(out)
        out = self.conv10(out)
        out = self.conv11(out)

        out = self.conv12(out)
        out = self.conv13(out)
        out = self.conv14(out)

        out = self.conv15(out)
        out = self.conv16(out)
        out = self.conv17(out)

        out = self.conv18(out)
        out = self.conv19(out)

        x = torch.flatten(out, start_dim=1)

        x = self.classifier(x)

        return x