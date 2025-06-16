import torch
import torch.nn as nn
import torch.nn.functional as F
from digicare.runtime.wmh_diagnosis.models._layer_presets import ConvBlock3D_DownSample, ConvBlock3D_Ordinary

class UNet_AttentionGate(nn.Module):
    def __init__(self, in_x_channels, in_gate_channels, intermediate_channels):
        super().__init__()
        self.phi_g = nn.Conv3d(in_gate_channels, intermediate_channels, 1, 1, 0)
        self.phi_x = nn.Conv3d(in_x_channels, intermediate_channels, 1, 2, 0)
        self.act_gx1 = nn.ReLU()
        self.phi_gx = nn.Conv3d(intermediate_channels, 1, 1, 1, 0)
        self.act_gx2 = nn.Sigmoid()
        self._cached_G = None # cached gate signal

    def forward(self, x: torch.Tensor, g: torch.Tensor):
        g0 = self.phi_g(g)
        x0 = self.phi_x(x)
        gx0 = g0 + x0
        gx1 = self.act_gx1(gx0)
        gx2 = self.phi_gx(gx1)
        gx3 = self.act_gx2(gx2)
        G: torch.Tensor = F.interpolate(gx3, scale_factor=2, mode='trilinear')
        self._cached_G = G.detach().cpu().numpy()
        return x * G

class UNet_Att_Cascade6_Cube128_Regression(nn.Module):
    '''
    UNet structure:
    1) 6 cascade resolution
    2) initial input volume shape: 128x128x128
    3) capable of performing image regression tasks
    4) with attention mechanism 
        Attention U-Net: https://arxiv.org/pdf/1804.03999.pdf
    '''
    def __init__(self, in_channels = None, out_channels = None, unit_width=16):
        super().__init__()
        fm=unit_width
        self.in_channels=in_channels
        self.cb_1_l=ConvBlock3D_Ordinary(in_channels,fm)
        self.g_1=UNet_AttentionGate(fm,2*fm,fm)
        self.cb_1_r=ConvBlock3D_Ordinary(2*fm,fm, use_norm=False)
        self.cb_2_l=ConvBlock3D_DownSample(fm,2*fm)
        self.g_2=UNet_AttentionGate(2*fm,4*fm,2*fm)
        self.cb_2_r=ConvBlock3D_Ordinary(4*fm,2*fm)
        self.cb_3_l=ConvBlock3D_DownSample(2*fm,4*fm)
        self.g_3=UNet_AttentionGate(4*fm,8*fm,4*fm)
        self.cb_3_r=ConvBlock3D_Ordinary(8*fm,4*fm)
        self.cb_4_l=ConvBlock3D_DownSample(4*fm,8*fm)
        self.g_4=UNet_AttentionGate(8*fm,16*fm,8*fm)
        self.cb_4_r=ConvBlock3D_Ordinary(16*fm,8*fm)
        self.cb_5_l=ConvBlock3D_DownSample(8*fm,16*fm)
        self.g_5=UNet_AttentionGate(16*fm,32*fm,16*fm)
        self.cb_5_r=ConvBlock3D_Ordinary(32*fm,16*fm)
        self.cb_6_l=ConvBlock3D_DownSample(16*fm,32*fm)
        self.cb_6_u=nn.ConvTranspose3d(32*fm,16*fm,2,2,0)
        self.cb_5_u=nn.ConvTranspose3d(16*fm,8*fm,2,2,0)
        self.cb_4_u=nn.ConvTranspose3d(8*fm,4*fm,2,2,0)
        self.cb_3_u=nn.ConvTranspose3d(4*fm,2*fm,2,2,0)
        self.cb_2_u=nn.ConvTranspose3d(2*fm,fm,2,2,0)

        self.fc_1=ConvBlock3D_Ordinary(fm, out_channels,use_nonlin=False,use_norm=False)

    def forward(self,x):
        assert len(x.shape) == 5, \
            'assume 5D tensor input with shape [b,c,x,y,z], got [%s].' % ','.join([str(s) for s in x.shape])
        assert all([x.shape[2] == 128, x.shape[3] == 128, x.shape[4] == 128]), \
            'input channel size should be 128*128*128, got %d*%d*%d.' % (x.shape[2],x.shape[3],x.shape[4])  
        assert x.shape[1] == self.in_channels, 'expected input to have %d channel(s), but got %d.' % x.shape[1]
        x1 = self.cb_1_l(x)
        x2 = self.cb_2_l(x1)
        x3 = self.cb_3_l(x2)
        x4 = self.cb_4_l(x3)
        x5 = self.cb_5_l(x4)
        x6 = self.cb_6_l(x5)
        x7 = self.cb_6_u(x6)
        x5g = self.g_5(x5,x6)
        x8 = torch.cat((x5g,x7),1)
        x9 = self.cb_5_r(x8)
        x10 = self.cb_5_u(x9)
        x4g = self.g_4(x4,x9)
        x11 = torch.cat((x4g,x10),1)
        x12 = self.cb_4_r(x11)
        x13 = self.cb_4_u(x12)
        x3g = self.g_3(x3,x12)
        x14 = torch.cat((x3g,x13),1)
        x15 = self.cb_3_r(x14)
        x16 = self.cb_3_u(x15)
        x2g = self.g_2(x2,x15)
        x17 = torch.cat((x2g,x16),1)
        x18 = self.cb_2_r(x17)
        x19 = self.cb_2_u(x18)
        x1g = self.g_1(x1,x18)
        x20 = torch.cat((x1g,x19),1)
        x21 = self.cb_1_r(x20)
        x22 = self.fc_1(x21)
        return x22

    def model_info(self):
        info = {}
        for name, parameter in self.named_parameters():
            info[name] = parameter
        return info

def model_benchmark(device='cuda:0'):
    batch_size = 1
    in_channels, out_channels = 1, 1
    unit_width = 16
    device_ = torch.device(device)
    input_shape = [batch_size,in_channels,128,128,128]
    x = torch.randn(input_shape).to(device_)
    model = UNet_Cascade6_Cube128_Regression(in_channels,out_channels,unit_width).to(device_)
    y: torch.Tensor = model(x)
    print('x:', x.shape)
    print('y:', y.shape)
    print('done.')

if __name__ == '__main__':
    model_benchmark()
