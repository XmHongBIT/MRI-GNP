import torch
import torch.nn as nn
'''
class Block_form(nn.Module):
    def __init__(self, in_channels, channels):
        super(Block_form, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.conva = nn.Conv3d(in_channels=self.in_channels, out_channels=self.channels, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm3d(self.channels)
        self.relua = nn.ReLU()
        self.convb = nn.Conv3d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, stride=1, padding=0)
        self.relub = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conva(x)
        x = self.relua(x)
        x = self.convb(x)
        x = self.relub(x)
        x = self.maxpool(x)
        return x

class Block_latt(nn.Module):
    def __init__(self, in_channels, channels):
        super(Block_latt, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.conva = nn.Conv3d(in_channels=self.in_channels, out_channels=self.channels, kernel_size=3, stride=1, padding=0)
        self.relua = nn.ReLU()
        self.convb = nn.Conv3d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, stride=1,
                               padding=0)
        self.relub = nn.ReLU()
        self.convc = nn.Conv3d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, stride=1,
                               padding=0)
        self.reluc = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conva(x)
        x = self.relua(x)
        x = self.convb(x)
        x = self.relub(x)
        x = self.convc(x)
        x = self.reluc(x)
        x = self.maxpool(x)
        return x

class VGG(nn.Module):
    def __init__(self, batch_size, output):
        super(VGG, self).__init__()
        self.batch_size = batch_size
        self.output = output
        self.block_f1 = Block_form(1, 64)
        self.block_f2 = Block_form(32, 32)
        self.block_l1 = Block_latt(32, 32)
        self.block_l2 = Block_latt(32, 64)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64, 128)
        self.bn = nn.BatchNorm1d(self.batch_size)
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, self.output)

    def forward(self, x):
        x = self.block_f1(x)
        x = self.block_f2(x)
        x = self.block_l1(x)
        x = self.block_l2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn(x.view(128, self.batch_size))
#        x = self.dropout(x.view(self.batch_size, 128))
        x = x.view(self.batch_size, 128)
        x = self.fc2(x)
        x = self.fc3(x)
        # x = nn.Softmax(dim=1)(x)
        return x
'''


class VGG_blocks(nn.Module):
    def __init__(self, channels, out_channels, maxpool):
        super(VGG_blocks, self).__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.maxpool1 = maxpool
        self.bn1 = nn.BatchNorm3d(self.channels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv3d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, stride=1,
                               padding=1)
        # self.dp1 = nn.Dropout3d(0.95)
        self.bn2 = nn.BatchNorm3d(self.channels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv3d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, stride=1,
                               padding=1)
        # self.dp2 = nn.Dropout3d(0.95)
        self.bn3 = nn.BatchNorm3d(self.channels)
        self.relu3 = nn.ReLU()
        self.conv3 = nn.Conv3d(in_channels=self.channels, out_channels=self.out_channels, kernel_size=3, stride=1,
                               padding=1)
        #self.dp3 = nn.Dropout3d(0.95)
        self.maxpool = nn.MaxPool3d(kernel_size=self.maxpool1, stride=2)

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        # x = self.dp1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        # x = self.dp2(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv3(x)
        #x = self.dp3(x)
        # x = self.maxpool(x)
        return x


class VGG_single_blocks(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGG_single_blocks, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bn = nn.BatchNorm3d(self.in_channels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv3d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        #self.dp = nn.Dropout3d(0.95)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        #x = self.dp(x)
        return x


class VGG(nn.Module):
    def __init__(self, batch_size, in_channels, channels, output, dropout):
        super(VGG, self).__init__()
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.channels = channels
        self.output = output
        self.dropout = dropout
        self.conv1 = nn.Conv3d(in_channels=self.in_channels, out_channels=self.channels, kernel_size=3, stride=1,
                               padding=1)
        # self.dpp = nn.Dropout3d(0.95)
        self.single_blk1 = VGG_single_blocks(self.channels, self.channels)
        self.maxpool1 = nn.MaxPool3d(2, 2)
        #self.single_blk2 = VGG_single_blocks(self.channels, self.channels*2)
        self.single_blk3 = VGG_single_blocks(self.channels, self.channels*2)
        self.maxpool2 = nn.MaxPool3d(2, 2)
        self.vgg_blok1 = VGG_blocks(self.channels*2, self.channels*4, 2)
        #self.vgg_blok2 = VGG_blocks(self.channels*4, self.channels*8, 2)
        self.maxpool3 = nn.MaxPool3d(2, 2)
        self.vgg_blok3 = VGG_blocks(self.channels*4, self.channels*8, 1)
        self.maxpool4 = nn.MaxPool3d(2, 2)
        # self.gap_fc = nn.Linear(channels*8, self.output)
        self.fc1 = nn.Linear(self.channels*8*8*8*8, 4096)
        self.dropout1 = nn.Dropout(self.dropout)
        self.fc2 = nn.Linear(4096, 4096)
        self.dropout2 = nn.Dropout(self.dropout)
        self.fc3 = nn.Linear(4096, self.output)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.dpp(x)
        x = self.single_blk1(x)
        x = self.maxpool1(x)
        #x = self.single_blk2(x)
        x = self.single_blk3(x)
        x = self.maxpool2(x)
        x = self.vgg_blok1(x)
        #x = self.vgg_blok2(x)
        x = self.maxpool3(x)
        x = self.vgg_blok3(x)
        x = self.maxpool4(x)
        # x = torch.nn.functional.adaptive_avg_pool3d(x, (1, 1, 1))
        x = x.view(self.batch_size, self.channels*8*8*8*8)
        # x = self.gap_fc(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    vgg = VGG(2, 4, 64, 1024, 0.1).to('cuda:2')
    input_data = torch.ones(2, 4, 61, 61, 61).to('cuda:2')
    output = vgg(input_data)
    print(output.shape)
