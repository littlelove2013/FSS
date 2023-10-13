import torch
from torch import nn
import torch.nn.functional as F

def vgg_block(num_convs, in_channels, out_channels):
    net = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
           nn.ReLU(True),
           nn.BatchNorm2d(out_channels)]
    for i in range(num_convs - 1): # 定义后面的许多层
        net.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        net.append(nn.ReLU(True))
        net.append(nn.BatchNorm2d(out_channels))
    net.append(nn.MaxPool2d(2, 2)) # 定义池化层
    return nn.Sequential(*net)
 
# 下面我们定义一个函数对这个 vgg block 进行堆叠
def vgg_stack(num_convs, channels): # vgg_net = vgg_stack((1, 1, 2, 2, 2), ((3, 64), (64, 128), (128, 256), (256, 512), (512, 512)))
    net = []
    for n, c in zip(num_convs, channels):
        in_c = c[0]
        out_c = c[1]
        net.append(vgg_block(n, in_c, out_c))
    return nn.Sequential(*net)
 
#确定vgg的类型，是vgg11 还是vgg16还是vgg19
# vgg_net = vgg_stack((1, 1, 2, 2, 2), ((3, 64), (64, 128), (128, 256), (256, 512), (512, 512)))
#vgg类
# VGG7_64((2,2,2),((3,64),(64,128),(128,256)))
class VGG(nn.Module):
    def __init__(self,num_convs,channels,out_dim=16):
        super().__init__()
        self.feature = vgg_stack(num_convs,channels)
        self.fc = nn.Sequential(
            nn.Linear(out_dim*channels[-1][1], 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 10))
        
    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
    
class VGG_plane(nn.Module):
    def __init__(self,num_convs,channels):
        super().__init__()
        self.feature = vgg_stack(num_convs,channels)
        self.avg_pool = nn.AvgPool2d(4)
        self.linear = nn.Linear(channels[-1][1], 10)
        
    def forward(self, x):
        x = self.feature(x)
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x

class MTL_VGG7_64_plane_base(nn.Module):
    def __init__(self,num_classes=10,dns_ratio=0.5):
        super().__init__()
        self.dns_ratio = dns_ratio
        # block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.act1 = nn.ReLU(True)
        self.bn1 = nn.BatchNorm2d(64)
        self.classifer1 = nn.Sequential(
                    nn.AvgPool2d(32),
                    nn.Flatten(),
                    nn.Linear(64, num_classes)
                )

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.act2 = nn.ReLU(True)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.classifer2 = nn.Sequential(
                    nn.AvgPool2d(16),
                    nn.Flatten(),
                    nn.Linear(64, num_classes)
                )

        # block 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.act3 = nn.ReLU(True)
        self.bn3 = nn.BatchNorm2d(128)
        self.classifer3 = nn.Sequential(
                    nn.AvgPool2d(16),
                    nn.Flatten(),
                    nn.Linear(128, num_classes)
                )
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.act4 = nn.ReLU(True)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.classifer4 = nn.Sequential(
                    nn.AvgPool2d(8),
                    nn.Flatten(),
                    nn.Linear(128, num_classes)
                )
        

        # block 3
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.act5 = nn.ReLU(True)
        self.bn5 = nn.BatchNorm2d(256)
        self.classifer5 = nn.Sequential(
                    nn.AvgPool2d(8),
                    nn.Flatten(),
                    nn.Linear(256, num_classes)
                )
        
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.act6 = nn.ReLU(True)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool6 = nn.MaxPool2d(2, 2)
        # classifier
        self.classifer6 = nn.Sequential(
            nn.AvgPool2d(4),
            nn.Flatten(),
            # nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        f1 = self.bn1(self.act1(self.conv1(x)))
        c1 = self.classifer1(f1)
        f2 = self.pool2(self.bn2(self.act2(self.conv2(f1))))
        c2 = self.classifer2(f2)
        f3 = self.bn3(self.act3(self.conv3(f2)))
        c3 = self.classifer3(f3)
        f4 = self.pool4(self.bn4(self.act4(self.conv4(f3))))
        c4 = self.classifer4(f4)
        f5 = self.bn5(self.act5(self.conv5(f4)))
        c5 = self.classifer5(f5)
        f6 = self.pool6(self.bn6(self.act6(self.conv6(f5))))
        c6 = self.classifer6(f6)
        return [c1,c2,c3,c4,c5,c6],[f1,f2,f3,f4,f5,f6]

class STL_VGG7_64_plane(MTL_VGG7_64_plane_base):
    def forward(self, x):
        logits,features = super().forward(x)
        return logits[-1]

def vgg64_7_plane_mtl(dns_ratio=0.5):
    return MTL_VGG7_64_plane_base(num_classes=100,dns_ratio=dns_ratio)

    
def vgg7_M(M):
    return VGG((2,2,2),((3,M),(M,2*M),(2*M,4*M)))

def vgg7_M_plane(M):
    return VGG_plane((2,2,2),((3,M),(M,2*M),(2*M,4*M)))
    
def vgg7_64():
    return vgg7_M(64)
    
def vgg7_64_plane():
    return vgg7_M_plane(64)