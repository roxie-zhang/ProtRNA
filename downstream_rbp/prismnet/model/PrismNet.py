import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import ResidualBlock1D, ResidualBlock2D
from .se import SEBlock

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        p0 = int((kernel_size[0] - 1) / 2) if same_padding else 0
        p1 = int((kernel_size[1] - 1) / 2) if same_padding else 0
        padding = (p0, p1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class PrismNet(nn.Module):
    def __init__(self, mode="pu"):
        super(PrismNet, self).__init__()
        self.mode = mode
        h_k = 5  #kernel height
        h_p = int((h_k - 1) / 2)
        if mode=="pu":
            self.n_features = 5
        elif mode=="seq":
            self.n_features = 4
            h_p, h_k = 1, 3 
        elif mode=="str":
            self.n_features = 1
            h_p, h_k = 0, 1
        elif mode=="lm640":
            self.n_features = 4 + 640 # for RNA-FM
        elif mode=="lm1280":
            self.n_features = 4 + 1280 # for ProtRNA and RiNALMo 
        # dimension reduction of lm features
        elif mode=="lm640_red":
            self.n_features = 16 + 32
            self.fc_seq_to_16 = nn.Linear(4, 16)
            self.fc_lm640_to_32 = nn.Linear(640, 32)
        elif mode=="lm768_red":
            self.n_features = 16 + 32
            self.fc_seq_to_16 = nn.Linear(4, 16)
            self.fc_lm768_to_32 = nn.Linear(768, 32)
        elif mode in ["lm1280_red", "rinalmo_red"]:
            self.n_features = 16 + 32   # 1:2
            self.fc_seq_to_16 = nn.Linear(4, 16)
            self.fc_lm1280_to_32 = nn.Linear(1280, 32)
        else:
            raise "mode error"
        
        base_channel = 8
        self.conv    = Conv2d(1, base_channel, kernel_size=(11, h_k), bn = True, same_padding=True)
        self.se      = SEBlock(base_channel)
        self.res2d   = ResidualBlock2D(base_channel, kernel_size=(11, h_k), padding=(5, h_p)) 
        self.res1d   = ResidualBlock1D(base_channel*4) 
        self.avgpool = nn.AvgPool2d((1,self.n_features))
        self.gpool   = nn.AdaptiveAvgPool1d(1)
        self.fc      = nn.Linear(base_channel*4*8, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, input):
        """[forward]
        
        Args:
            input ([tensor],N,C,W,H): input features
        """
        if self.mode=="seq":
            input = input[:,:,:,:4]
        elif self.mode=="str":
            input = input[:,:,:,4:]
        elif self.mode=="lm640_red":
            seq4, lm640 = input[:, :, :, :4], input[:, :, :, 4:]
            seq16 = self.fc_seq_to_16(seq4)
            lm32 = self.fc_lm640_to_32(lm640)
            input = torch.cat((seq16, lm32), dim=3)
        elif self.mode=="lm768_red":
            seq4, lm768 = input[:, :, :, :4], input[:, :, :, 4:]
            seq16 = self.fc_seq_to_16(seq4)
            lm32 = self.fc_lm768_to_32(lm768)
            input = torch.cat((seq16, lm32), dim=3)
        elif self.mode in ["lm1280_red", "rinalmo_red"]:
            seq4, lm1280 = input[:, :, :, :4], input[:, :, :, 4:]
            seq16 = self.fc_seq_to_16(seq4)
            lm32 = self.fc_lm1280_to_32(lm1280)
            input = torch.cat((seq16, lm32), dim=3)
        elif self.mode=="lm1280_red128":
            seq4, lm1280 = input[:, :, :, :4], input[:, :, :, 4:]
            seq16 = self.fc_seq_to_16(seq4)
            lm128 = self.fc_lm1280_to_128(lm1280)
            input = torch.cat((seq16, lm128), dim=3)
        elif self.mode=="lm1280_red16":
            seq4, lm1280 = input[:, :, :, :4], input[:, :, :, 4:]
            lm16 = self.fc_lm1280_to_16(lm1280)
            input = torch.cat((seq4, lm16), dim=3)

        x = self.conv(input)
        x = F.dropout(x, 0.1, training=self.training)
        z = self.se(x)
        x = self.res2d(x*z)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.avgpool(x)
        x = x.view(x.shape[0], x.shape[1], x.shape[2])
        x = self.res1d(x)
        x = F.dropout(x, 0.3, training=self.training)
        x = self.gpool(x)
        x = x.view(x.shape[0], x.shape[1])
        x = self.fc(x)
        return x


class PrismNet_large(nn.Module):
    def __init__(self, mode="pu"):
        super(PrismNet_large, self).__init__()
        self.mode = mode
        h_p, h_k = 2, 5 
        if mode=="pu":
            self.n_features = 5
        elif mode=="seq":
            self.n_features = 4
            h_p, h_k = 1, 3 
        elif mode=="str":
            self.n_features = 1
            h_p, h_k = 0, 1
        else:
            raise "mode error"
        
        base_channel = 64
        self.conv    = Conv2d(1, base_channel, kernel_size=(11, h_k), bn = True, same_padding=True)
        self.se      = SEBlock(base_channel)
        self.res2d   = ResidualBlock2D(base_channel, kernel_size=(11, h_k), padding=(5, h_p)) 
        self.res1d   = ResidualBlock1D(base_channel*4) 
        self.avgpool = nn.AvgPool2d((1,self.n_features))
        self.gpool   = nn.AdaptiveAvgPool1d(1)
        self.fc      = nn.Linear(base_channel*4*8, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, input):
        """[summary]
        
        Args:
            input ([tensor],N,C,W,H): input features
        """
        if self.mode=="seq":
            input = input[:,:,:,:4]
        elif self.mode=="str":
            input = input[:,:,:,4:]
        x = self.conv(input)
        x = F.dropout(x, 0.1, training=self.training)
        z = self.se(x)
        x = self.res2d(x*z)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.avgpool(x)
        x = x.view(x.shape[0], x.shape[1], x.shape[2])
        x = self.res1d(x)
        x = F.dropout(x, 0.3, training=self.training)
        x = self.gpool(x)
        x = x.view(x.shape[0], x.shape[1])
        x = self.fc(x)
        return x
