import torch
import torch.nn as nn

# Conv Layer
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

# Upsample Conv Layer
class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample = nn.Upsample(scale_factor=upsample, mode='nearest')
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = self.upsample(x)
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out 

# CyclGAN generator 
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # nonlinearities
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # c7s1-32
        self.conv1_e = ConvLayer(3, 32, kernel_size=7, stride=1)
        self.in1_e = nn.InstanceNorm2d(32, affine=True)

        # d64
        self.conv2_e = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2_e = nn.InstanceNorm2d(64, affine=True)

        # d128
        self.conv3_e = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3_e = nn.InstanceNorm2d(128, affine=True)

        # residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        self.res6 = ResidualBlock(128)
        self.res7 = ResidualBlock(128)
        self.res8 = ResidualBlock(128)
        self.res9 = ResidualBlock(128)

        #u64
        self.conv1_d = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in1_d = nn.InstanceNorm2d(64, affine=True)

        #u32
        self.conv2_d = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in2_d = nn.InstanceNorm2d(32, affine=True)
        
        #c7s1-3
        self.conv3_d = ConvLayer(32, 3, kernel_size=7, stride=1)
        self.in3_d = nn.InstanceNorm2d(3, affine=True)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)


    def forward(self, x):
        # encode
        y = self.relu(self.in1_e(self.conv1_e(x)))
        y = self.relu(self.in2_e(self.conv2_e(y)))
        y = self.relu(self.in3_e(self.conv3_e(y)))

        # residual
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.res6(y)
        y = self.res7(y)
        y = self.res8(y)
        y = self.res9(y)

        # decode
        y = self.relu(self.in1_d(self.conv1_d(y)))
        y = self.relu(self.in2_d(self.conv2_d(y)))
        y = self.tanh(self.in3_d(self.conv3_d(y)))
        #y = self.conv3_d(y)

        return y

# PatchGAN
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # nonlinearities
        self.lrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

        # C64
        self.conv1 = ConvLayer(3, 64, kernel_size=4, stride=2)

        # C128
        self.conv2 = ConvLayer(64, 128, kernel_size=4, stride=2)
        self.in2 = nn.InstanceNorm2d(128, affine=True)

        # 256
        self.conv3 = ConvLayer(128, 256, kernel_size=4, stride=2)
        self.in3 = nn.InstanceNorm2d(256, affine=True)

        # 512 (stride 1 for receptive field to be 70 x 70)
        self.conv4 = ConvLayer(256, 512, kernel_size=4, stride=1)
        self.in4 = nn.InstanceNorm2d(512, affine=True)

        # 1-d output
        self.conv5 = ConvLayer(512, 1, kernel_size=4, stride=1)
        #self.in5 = nn.InstanceNorm2d(1, affine=True)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)


    def forward(self, x):
        y = self.lrelu(self.conv1(x))
        y = self.lrelu(self.in2(self.conv2(y)))
        y = self.lrelu(self.in3(self.conv3(y)))
        y = self.lrelu(self.in4(self.conv4(y)))
        y = self.sigmoid(self.conv5(y))
        return y

