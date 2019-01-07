import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class BottleNeck(nn.Module):
    """ResNet BottleNeck"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(BottleNeck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2  = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
                )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        sh = self.shortcut(x)
        out += sh
        out = F.relu(out)
        return out

class FPN(nn.Module):
    """FPN"""
    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,  bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-Up
        self.stage1 = self._make_stage(BottleNeck, 64, num_blocks[0], stride=1)
        self.stage2 = self._make_stage(BottleNeck, 128, num_blocks[1], stride=2)
        self.stage3 = self._make_stage(BottleNeck, 256, num_blocks[2], stride=2)
        self.stage4 = self._make_stage(BottleNeck, 512, num_blocks[3], stride=2)

        # Top-Down
        self.top_stage = nn.Conv2d(self.in_planes, 256, kernel_size=1, stride=1, padding=0) # Reduce Channel
        self.leteral1 = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=0)
        self.leteral2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.leteral3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)

        # Smooth
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    
    def _make_stage(self, block, planes, num_blocks, stride):
        """make Bottom-Up stage"""
        strides =  [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        """upsample and add"""
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        # Bottom-UP
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=2, stride=2, padding=1)
        c2 = self.stage1(c1)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        # Top-Down
        p5 = self.top_stage(c5)
        p4 = self._upsample_add(p5, self.leteral1(c4))
        p3 = self._upsample_add(p4, self.leteral2(c3))
        p2 = self._upsample_add(p3, self.leteral3(c2))
        # Smooth
        p5 = self.smooth1(p5)
        p4 = self.smooth2(p4)
        p3 = self.smooth3(p3)

        return p2, p3, p4, p5
    
def FPN101():
    return FPN(BottleNeck, [2,2,2,2])


def test():
    net = FPN101()
    fms = net(Variable(torch.randn(1,3,600,900)))
    for fm in fms:
        print(fm.size())

if __name__ == "__main__":
    test()
