import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_channel, out_channel, stride, padding):
# 3*3 convolution
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=padding, dilation =1, groups=1, bias=False)


def conv1x1(in_channel, out_channel, stride, padding):
# 1*1 convolution
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=padding, dilation =1, groups=1, bias=False)

class basic_block(nn.Module):
    def __init__(self, inplanes, planes, stride=1, groups=1, down=None):
        super().__init__()
        if down is not None:
            # stride 값을 늘려서 downsampling 시킴
            self.conv1=conv3x3(inplanes,planes,stride=2, padding=1)  
        else:
            self.conv1= conv3x3(inplanes, planes, stride=1, padding=1) 
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes,planes,stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu=nn.ReLU(inplace = True)
        if planes==16:
            self.downsample = nn.Conv2d(inplanes,planes,kernel_size=3, stride=1, padding=1)
        else:
            self.downsample = nn.Conv2d(inplanes,planes,kernel_size=3, stride=2, padding=1)
        self.down =down
    
    def forward(self,x):
        ## 정의한 layer를 진행시킨다
        residual=x
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.conv2(out)
        out=self.bn2(out)
        if self.down is not None:
            residual=self.downsample(residual)
        out=out+residual
        out=self.relu(out)

        return out

class bottleneck_block(nn.Module):
    def __init__(self, inplanes, planes, stride=1, groups=1, down=None):
        super().__init__()
        if self.down is not None:
            # stride 값을 늘려서 downsampling 시킴
            self.conv1=conv1x1(inplanes,planes,stride=2, padding=1)  
        else:
            self.conv1= conv1x1(inplanes, planes, stride=1, padding=1) 
        
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = conv3x3(planes,planes,stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = conv1x1(planes,planes,stride=1, padding=1)
        self.bn3= nn.BatchNorm2d(256)
        self.relu= nn.ReLU(inplace=True)  #inplace=true -> the input is modified directly
        


    def forward(self, x):
          ## 정의한 layer를 진행시킨다.
        residual =x 
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu(out)
        out=self.conv3(out)
        out=self.bn3(out)

        if self.down is not None:
            residual=self.downsample(residual)
        out=out+residual
        out=self.relu(out)

       
        return out

    

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, zero_init_residual=True):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, dilation =1, groups=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True) 
        self.layer1 = self.make_layer(block, 16, 16, num_blocks[0])
        self.layer2 = self.make_layer(block, 16, 32, num_blocks[1])
        self.layer3 = self.make_layer(block, 32, 64, num_blocks[2])
        self.avgpool = nn.AvgPool2d(2,stride=2)
        self.linear = nn.Linear(1024, 10)

        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, bottleneck_block):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, basic_block):
                    nn.init.constant_(m.bn2.weight, 0)
                    




    def make_layer(self, block, inplanes, planes, num_block): 
        layers = []
        if planes==16:
            tmp=None
        else:
            tmp=1
        layers.append(basic_block(inplanes,planes, down=tmp))
        for i in range(1,num_block):
            layers.append(basic_block(planes,planes,down=None))
        
        return nn.Sequential(*layers)


    def forward(self, x):
        out=self.conv1(x)
        #print(out.size())
        out=self.bn1(out)
        #print(out.size())
        out=self.relu(out)
        #print(out.size())
        out=self.layer1(out)
        #print(out.size())
        out=self.layer2(out)
        #print(out.size())
        out=self.layer3(out)
        #print(out.size())
        out = self.avgpool(out)
        #print(out.size())
        out = Tensor.flatten(out,1)
        #print(out.size())
        out = self.linear(out)
        #print(out.size())
        #out = F.softmax(out, dim=1)
        #print(out.size())
        return out

def ResNet20():
    return ResNet(basic_block, [7, 6, 6])
def ResNet32():
    return ResNet(basic_block, [11, 10, 10])
def ResNet44():
    return ResNet(basic_block, [15, 14, 14])
def ResNet56():
    return ResNet(basic_block, [19, 18, 18])
def ResNet110():
    return ResNet(basic_block, [37, 36, 36])
