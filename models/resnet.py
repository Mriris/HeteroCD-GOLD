import math
import os
import sys
import torch.fft
import torch
import torch.nn as nn

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

__all__ = ['resnet18', 'resnet50', 'resnet101']

base_url = 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/'
model_urls = {
    'resnet18': base_url + 'resnet18-imagenet.pth',
    'resnet50': base_url + 'resnet50-imagenet.pth',
    'resnet101': base_url + 'resnet101-imagenet.pth'
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Convkxk(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=1,
                 stride=1,
                 padding=0):
        super(Convkxk, self).__init__()
        self.conv = nn.Conv2d(in_planes,
                              out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class FrequencySeparation(nn.Module):
    def __init__(self):
        super(FrequencySeparation, self).__init__()

    def forward(self, x, return_type='both'):
        """
        x: 输入特征图，尺寸为 (B, C, H, W)
        return_type: 指定返回类型，'low' 返回低频，'high' 返回高频，'both' 返回两者
        """
        # 1. 输入特征图的尺寸 (B, C, H, W)
        B, C, H, W = x.shape

        # 2. 对特征图进行2D傅里叶变换，输出尺寸与输入一致 (B, C, H, W)
        fft_x = torch.fft.fft2(x)

        # 3. 将频率域结果移位，低频成分移动到中心，尺寸不变 (B, C, H, W)
        fft_x_shifted = torch.fft.fftshift(fft_x)

        # 4. 获取中心位置和低频掩码的半径
        center_h, center_w = H // 2, W // 2
        radius_h, radius_w = H // 4, W // 4  # 半径

        # 根据 return_type 决定需要计算的内容
        if return_type == 'low':
            # 构造低频掩码
            low_freq_mask = torch.zeros_like(fft_x)
            low_freq_mask[..., center_h - radius_h:center_h + radius_h, center_w - radius_w:center_w + radius_w] = 1

            # 低频部分
            low_freq_component = fft_x_shifted * low_freq_mask

            # 将低频部分转换回空间域
            low_freq_back = torch.fft.ifftshift(low_freq_component)
            low_freq_img = torch.fft.ifft2(low_freq_back).real

            return low_freq_img

        elif return_type == 'high':
            # 构造高频掩码
            high_freq_mask = torch.ones_like(fft_x)
            high_freq_mask[..., center_h - radius_h:center_h + radius_h, center_w - radius_w:center_w + radius_w] = 0

            # 高频部分
            high_freq_component = fft_x_shifted * high_freq_mask

            # 将高频部分转换回空间域
            high_freq_back = torch.fft.ifftshift(high_freq_component)
            high_freq_img = torch.fft.ifft2(high_freq_back).real

            return high_freq_img

        else:  # 默认返回高频和低频
            # 构造低频掩码
            low_freq_mask = torch.zeros_like(fft_x)
            low_freq_mask[..., center_h - radius_h:center_h + radius_h, center_w - radius_w:center_w + radius_w] = 1

            # 构造高频掩码
            high_freq_mask = 1 - low_freq_mask

            # 低频部分
            low_freq_component = fft_x_shifted * low_freq_mask
            low_freq_back = torch.fft.ifftshift(low_freq_component)
            low_freq_img = torch.fft.ifft2(low_freq_back).real

            # 高频部分
            high_freq_component = fft_x_shifted * high_freq_mask
            high_freq_back = torch.fft.ifftshift(high_freq_component)
            high_freq_img = torch.fft.ifft2(high_freq_back).real

            return low_freq_img, high_freq_img

def adain(x1, x2, eps=1e-5):
    """
    将风格特征 x1 注入到内容特征 x 中.
    
    参数:
    x  (torch.Tensor): 内容特征，形状为 (batch_size, channels, height, width)
    x1 (torch.Tensor): 风格特征，形状为 (batch_size, channels, height, width)
    eps (float): 防止除零的小值

    返回:
    torch.Tensor: 融合后的特征图
    """
    
    # 计算内容特征的均值和标准差 (逐通道计算)
    # print(x1.shape)
    mean_x1 = torch.mean(x1, dim=(2, 3), keepdim=True)  # (batch_size, channels, 1, 1)
    std_x1 = torch.std(x1, dim=(2, 3), keepdim=True) + eps  # (batch_size, channels, 1, 1)
    # 计算风格特征的均值和标准差 (逐通道计算)
    mean_x2 = torch.mean(x2, dim=(2, 3), keepdim=True)  # (batch_size, channels, 1, 1)
    std_x2 = torch.std(x2, dim=(2, 3), keepdim=True) + eps  # (batch_size, channels, 1, 1)
    # 归一化内容特征 (去均值并归一化标准差)
    normalized_x = (x1 - mean_x1) / std_x1    
    # 调整内容特征的均值和标准差为风格特征的均值和标准差
    out = normalized_x * std_x2 + mean_x2
    return out
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, opt_backbone=False):
        super(ResNet, self).__init__()
        self.inplanes = 128
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.opt_backbone = opt_backbone
        self.FrequencySeparation = FrequencySeparation()
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    
    def forward(self, x,  x_sar=None):
        #if (self.opt_backbone==False):
        if (2>1):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            f = []
            f.append(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            f.append(x)
            x = self.layer2(x)
            f.append(x)
            x = self.layer3(x)
            f.append(x)
            x = self.layer4(x)
            f.append(x)
        else:
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            f = []
            f.append(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.FrequencySeparation(x, return_type='low')
            # x_sar0 = self.FrequencySeparation(x_sar[0], return_type='high')
            
            x = adain(x, x_sar[0])
            f.append(x)
            # f_sar.append(adain(self.FrequencySeparation(x_sar[0], return_type='low'),x_sar0))
            # 
            # print("-----------------------------------------------------------------------")
            x = self.layer2(x)
            x = self.FrequencySeparation(x, return_type='low')
            # x_sar1 = self.FrequencySeparation(x_sar[1], return_type='high')
            x = adain(x, x_sar[1])
            f.append(x)
            # f_sar.append(adain(self.FrequencySeparation(x_sar[1], return_type='low'),x_sar1))
            x = self.layer3(x)
            x = self.FrequencySeparation(x, return_type='low')
            # x_sar2 = self.FrequencySeparation(x_sar[2], return_type='high')
            x = adain(x, x_sar[2])
            f.append(x)
            # f_sar.append(adain(self.FrequencySeparation(x_sar[2], return_type='low'),x_sar2))
            x = self.layer4(x)
            x = self.FrequencySeparation(x, return_type='low')
            # x_sar3 = self.FrequencySeparation(x_sar[3], return_type='high')
            x = adain(x, x_sar[3])
            # f_sar.append(adain(self.FrequencySeparation(x_sar[3], return_type='low'),x_sar3))
            f.append(x)
            # print("-----------------------------------------------------------------------")
            # return tuple(f), tuple(f_sar)
        return tuple(f)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        # return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on Places
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet18']), strict=False)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on Places
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet50']), strict=False)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on Places
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet101']), strict=False)
    return model


def load_url(url, model_dir='./pretrained', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)

if __name__ == '__main__':
    model = resnet18(pretrained=True)
    data = torch.randn(1, 3, 512, 512)
    output = model(data)
    print(output[0].shape, output[1].shape, output[2].shape, output[3].shape, output[4].shape)