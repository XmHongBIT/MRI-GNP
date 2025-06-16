# define basic models and building blocks
import torch
import torch.nn as nn
from digicare.runtime.wmh_diagnosis.models._layer_presets import ConvBlock3D_DownSample, ConvBlock3D_Ordinary

class UNet_6Res_DeepSup(nn.Module):
    '''
    UNet structure:
    1) 6 cascade resolution, initial input volume shape: 128x128x128
    2) image regression branch
    3) classification branch
    4) deep supervision
    '''
    def __init__(self, in_channels = None, 
        out_reg_channels = None, 
        out_cls_channels = None, unit_width=16):
        super().__init__()
        fm=unit_width
        self.in_channels=in_channels
        self.out_reg_channels=out_reg_channels
        self.out_cls_channels=out_cls_channels
        self.cb_1_l=ConvBlock3D_Ordinary(in_channels,fm)
        self.cb_1_r=ConvBlock3D_Ordinary(2*fm,fm)
        self.cb_2_l=ConvBlock3D_DownSample(fm,2*fm)
        self.cb_2_r=ConvBlock3D_Ordinary(4*fm,2*fm)
        self.cb_3_l=ConvBlock3D_DownSample(2*fm,4*fm)
        self.cb_3_r=ConvBlock3D_Ordinary(8*fm,4*fm)
        self.cb_4_l=ConvBlock3D_DownSample(4*fm,8*fm)
        self.cb_4_r=ConvBlock3D_Ordinary(16*fm,8*fm)
        self.cb_5_l=ConvBlock3D_DownSample(8*fm,16*fm)
        self.cb_5_r=ConvBlock3D_Ordinary(32*fm,16*fm)
        self.cb_6_l=ConvBlock3D_DownSample(16*fm,32*fm)
        self.cb_6_u=nn.ConvTranspose3d(32*fm,16*fm,2,2,0)
        self.cb_5_u=nn.ConvTranspose3d(16*fm,8*fm,2,2,0)
        self.cb_4_u=nn.ConvTranspose3d(8*fm,4*fm,2,2,0)
        self.cb_3_u=nn.ConvTranspose3d(4*fm,2*fm,2,2,0)
        self.cb_2_u=nn.ConvTranspose3d(2*fm,fm,2,2,0)

        self.fc_1=nn.Conv3d(fm,out_reg_channels,3,1,1)
        self.fc_2=nn.Conv3d(2*fm,out_reg_channels,3,1,1)
        self.fc_3=nn.Conv3d(4*fm,out_reg_channels,3,1,1)
        self.fc_4=nn.Conv3d(8*fm,out_reg_channels,3,1,1)
        self.fc_5=nn.Conv3d(16*fm,out_reg_channels,3,1,1)

        self.fc_6_a=nn.Sequential(
            nn.Conv3d(32*fm,64*fm,2,2,0),
            nn.InstanceNorm3d(64*fm),
            nn.LeakyReLU(),
            nn.Conv3d(64*fm,128*fm,2,2,0), 
            # after this layer, spatial size=1, so don't need instance norm.
            nn.LeakyReLU())
        self.fc_6_b = nn.Linear(128*fm,out_cls_channels)
        
    def forward(self,x):
        assert len(x.shape) == 5, \
            'assume 5D tensor input with shape [b,c,x,y,z], got [%s].' % ','.join([str(s) for s in x.shape])
        assert all([x.shape[2] == 128, x.shape[3] == 128, x.shape[4] == 128]), \
            'input channel size should be 128*128*128, got %d*%d*%d.' % (x.shape[2],x.shape[3],x.shape[4])  
        assert x.shape[1] == self.in_channels, 'expected input to have %d channel(s), but got %d.' % x.shape[1]
        batch_size = x.shape[0]
        x1 = self.cb_1_l(x)
        x2 = self.cb_2_l(x1)
        x3 = self.cb_3_l(x2)
        x4 = self.cb_4_l(x3)
        x5 = self.cb_5_l(x4)
        x6 = self.cb_6_l(x5)
        x7 = self.cb_6_u(x6)
        x8 = torch.cat((x5,x7),1)
        x9 = self.cb_5_r(x8)
        x10 = self.cb_5_u(x9)
        x11 = torch.cat((x4,x10),1)
        x12 = self.cb_4_r(x11)
        x13 = self.cb_4_u(x12)
        x14 = torch.cat((x3,x13),1)
        x15 = self.cb_3_r(x14)
        x16 = self.cb_3_u(x15)
        x17 = torch.cat((x2,x16),1)
        x18 = self.cb_2_r(x17)
        x19 = self.cb_2_u(x18)
        x20 = torch.cat((x1,x19),1)
        x21 = self.cb_1_r(x20)
        y1 = self.fc_1(x21)
        y2 = self.fc_2(x18)
        y3 = self.fc_3(x15)
        y4 = self.fc_4(x12)
        y5 = self.fc_5(x9)
        c = self.fc_6_a(x6)
        assert all([c.shape[2] == 1, c.shape[3] == 1, c.shape[4] == 1]), \
            'expected output logits to have spatial shape 1*1*1, but got %d*%d*%d.' % \
            (c.shape[2],c.shape[3],c.shape[4])
        c = torch.reshape(c, [batch_size, c.shape[1]])
        c = self.fc_6_b(c)
        return c, y1, y2, y3, y4, y5

    def model_info(self):
        info = {}
        for name, parameter in self.named_parameters():
            info[name] = parameter
        return info

def model_benchmark(model_config={'unit_width':16}, device='cuda:0'):
    input_shape = [1,1,128,128,128]
    device_ = torch.device(device)
    x = torch.randn(input_shape).to(device_)
    out_classes = 2
    out_image_channels = 1
    model = UNet_6Res_DeepSup(1,out_image_channels,out_classes,model_config['unit_width']).to(device_)
    out = model(x)
    print('x:', x.shape)
    for name, item in zip(['c', 'y1', 'y2', 'y3', 'y4'], out):
        print(name+':', item.shape)
    print('done.')

if __name__ == '__main__':
    model_benchmark()