import torch
import torch.nn as nn

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


        self.conv11 = Conv(in_ch, 64, 3, 1, 1)      
        self.conv12 = Conv(64, 64, 3, 1, 1)        
        self.down1 = nn.MaxPool1d(2)                

        self.conv21 = Conv(64, 128, 3, 1, 1)       
        self.conv22 = Conv(128, 128, 3, 1, 1)      
        self.down2 = nn.MaxPool1d(2)                

        self.conv31 = Conv(128, 256, 3, 1, 1)     
        self.conv32 = Conv(256, 256, 3, 1, 1)       
        self.conv33 = Conv(256, 256, 3, 1, 1)       


        self.up1 = nn.ConvTranspose1d(256, 256, 2, 2, )    
        self.conv41 = Conv(384, 128, 3, 1, 1)              
        self.conv42 = Conv(128, 128, 3, 1, 1)              

        self.up2 = nn.ConvTranspose1d(128, 128, 2, 2)        
        self.conv51 = Conv(192, 64, 3, 1, 1)               
        self.conv52 = Conv(64, 64, 3, 1, 1)                 

        self.conv61 = nn.Conv1d(64, 1, 3, 1, 1)            


        self.classifier = nn.Sequential(
            nn.Linear(2 * 512, 2048),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, out_ch)
        )

    def forward(self, x):

        x1 = self.conv11(x)
        x2 = self.conv12(x1)
        x3 = self.down1(x2)

        x4 = self.conv21(x3)
        x5 = self.conv22(x4)
        x6 = self.down2(x5)

        x7 = self.conv31(x6)
        x8 = self.conv32(x7)
        x9 = self.conv33(x8)

        x10 = self.up1(x9)
        d1 = torch.cat((x5, x10), dim=1)
        x11 = self.conv41(d1)
        x12 = self.conv42(x11)

        x13 = self.up2(x12)
        d2 = torch.cat((x2, x13), dim=1)
        x14 = self.conv51(d2)
        x15 = self.conv52(x14)
        x16 = self.conv61(x15)

        d3 = torch.cat((x, x16), dim=1)

        x = torch.flatten(d3, start_dim=1)
        x = self.classifier(x)
        out = nn.ReLU()(x)

        return out



