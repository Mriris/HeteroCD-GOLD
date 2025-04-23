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

__all__ = ['resnet18', 'resnet50', 'resnet101', 'lightweight_resnet18']

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
    """
    定义 ResNet 中的基本残差块类。
    基本残差块由两个 3x3 卷积层组成，用于构建 ResNet-18 和 ResNet-34 等浅层网络。

    Attributes:
        expansion (int): 输出通道数相对于输入通道数的扩展因子，基本残差块中为 1。
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        初始化基本残差块。

        Args:
            inplanes (int): 输入特征图的通道数。
            planes (int): 卷积层的输出通道数。
            stride (int, 可选): 第一个卷积层的步长，默认为 1。
            downsample (nn.Module, 可选): 用于下采样的模块，当输入和输出的尺寸或通道数不匹配时使用，默认为 None。
        """
        super(BasicBlock, self).__init__()
        # 第一个 3x3 卷积层
        self.conv1 = conv3x3(inplanes, planes, stride)
        # 第一个批量归一化层
        self.bn1 = nn.BatchNorm2d(planes)
        # 激活函数 ReLU
        self.relu = nn.ReLU(inplace=True)
        # 第二个 3x3 卷积层
        self.conv2 = conv3x3(planes, planes)
        # 第二个批量归一化层
        self.bn2 = nn.BatchNorm2d(planes)
        # 下采样模块
        self.downsample = downsample
        # 第一个卷积层的步长
        self.stride = stride

    def forward(self, x):
        """
        定义基本残差块的前向传播过程。

        Args:
            x (torch.Tensor): 输入的特征图。

        Returns:
            torch.Tensor: 经过残差块处理后的特征图。
        """
        # 保存输入作为残差连接的输入
        residual = x

        # 第一个卷积层及其后的批量归一化和激活函数
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二个卷积层及其后的批量归一化
        out = self.conv2(out)
        out = self.bn2(out)

        # 如果存在下采样模块，则对输入进行下采样以匹配输出尺寸
        if self.downsample is not None:
            residual = self.downsample(x)

        # 将残差连接的结果加到输出上
        out += residual
        # 最后应用激活函数
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    定义 ResNet 中的瓶颈残差块类。
    瓶颈残差块由一个 1x1 卷积层、一个 3x3 卷积层和另一个 1x1 卷积层组成，
    用于构建 ResNet-50、ResNet-101 和 ResNet-152 等深层网络。

    Attributes:
        expansion (int): 输出通道数相对于中间卷积层输出通道数的扩展因子，瓶颈残差块中为 4。
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        初始化瓶颈残差块。

        Args:
            inplanes (int): 输入特征图的通道数。
            planes (int): 中间卷积层的输出通道数。
            stride (int, 可选): 3x3 卷积层的步长，默认为 1。
            downsample (nn.Module, 可选): 用于下采样的模块，当输入和输出的尺寸或通道数不匹配时使用，默认为 None。
        """
        super(Bottleneck, self).__init__()
        # 第一个 1x1 卷积层，用于减少通道数
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        # 第一个批量归一化层
        self.bn1 = nn.BatchNorm2d(planes)
        # 3x3 卷积层，可能进行下采样
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        # 第二个批量归一化层
        self.bn2 = nn.BatchNorm2d(planes)
        # 第二个 1x1 卷积层，用于增加通道数
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        # 第三个批量归一化层
        self.bn3 = nn.BatchNorm2d(planes * 4)
        # 激活函数 ReLU
        self.relu = nn.ReLU(inplace=True)
        # 下采样模块
        self.downsample = downsample
        # 3x3 卷积层的步长
        self.stride = stride

    def forward(self, x):
        """
        定义瓶颈残差块的前向传播过程。

        Args:
            x (torch.Tensor): 输入的特征图。

        Returns:
            torch.Tensor: 经过残差块处理后的特征图。
        """
        # 保存输入作为残差连接的输入
        residual = x

        # 第一个 1x1 卷积层及其后的批量归一化和激活函数
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 3x3 卷积层及其后的批量归一化和激活函数
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 第二个 1x1 卷积层及其后的批量归一化
        out = self.conv3(out)
        out = self.bn3(out)

        # 如果存在下采样模块，则对输入进行下采样以匹配输出尺寸
        if self.downsample is not None:
            residual = self.downsample(x)

        # 将残差连接的结果加到输出上
        out += residual
        # 最后应用激活函数
        out = self.relu(out)

        return out


class Convkxk(nn.Module):
    """
    定义一个自定义的卷积模块，包含卷积层、批量归一化层和 ReLU 激活函数。

    Args:
        in_planes (int): 输入特征图的通道数。
        out_planes (int): 输出特征图的通道数。
        kernel_size (int, 可选): 卷积核的大小，默认为 1。
        stride (int, 可选): 卷积的步长，默认为 1。
        padding (int, 可选): 卷积的填充数，默认为 0。
    """
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=1,
                 stride=1,
                 padding=0):
        # 调用父类的构造函数
        super(Convkxk, self).__init__()
        # 定义卷积层，不使用偏置项
        self.conv = nn.Conv2d(in_planes,
                              out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False)
        # 定义批量归一化层
        self.bn = nn.BatchNorm2d(out_planes)
        # 定义 ReLU 激活函数，inplace=True 表示直接在原张量上进行操作，节省内存
        self.relu = nn.ReLU(inplace=True)

        # 初始化模块的权重和偏置
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 计算卷积层的参数数量
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # 使用正态分布初始化卷积层的权重
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                # 初始化批量归一化层的权重为 1
                m.weight.data.fill_(1)
                # 初始化批量归一化层的偏置为 0
                m.bias.data.zero_()

    def forward(self, x):
        """
        定义模块的前向传播过程。

        Args:
            x (torch.Tensor): 输入的特征图。

        Returns:
            torch.Tensor: 经过卷积、批量归一化和 ReLU 激活函数处理后的特征图。
        """
        # 依次通过卷积层、批量归一化层和 ReLU 激活函数
        return self.relu(self.bn(self.conv(x)))


class FrequencySeparation(nn.Module):
    def __init__(self):
        # 调用父类的构造函数进行初始化
        super(FrequencySeparation, self).__init__()

    def forward(self, x, return_type='both'):
        """
        x: 输入特征图，尺寸为 (B, C, H, W)
        return_type: 指定返回类型，'low' 返回低频，'high' 返回高频，'both' 返回两者
        """
        # 1. 获取输入特征图的尺寸，B 为批量大小，C 为通道数，H 为高度，W 为宽度
        B, C, H, W = x.shape

        # 2. 对输入的特征图进行二维傅里叶变换，将其从空间域转换到频率域
        # 输出的频率域特征图尺寸与输入一致，为 (B, C, H, W)
        fft_x = torch.fft.fft2(x)

        # 3. 将频率域结果进行移位操作，把低频成分移动到特征图的中心位置
        # 移位后尺寸不变，仍为 (B, C, H, W)
        fft_x_shifted = torch.fft.fftshift(fft_x)

        # 4. 计算特征图的中心位置，以及低频掩码的半径
        center_h, center_w = H // 2, W // 2
        # 半径设置为高度和宽度的四分之一
        radius_h, radius_w = H // 4, W // 4

        # 根据 return_type 决定需要计算的内容
        if return_type == 'low':
            # 构造低频掩码，初始化为全零张量
            low_freq_mask = torch.zeros_like(fft_x)
            # 将中心区域设置为 1，该区域对应低频部分
            low_freq_mask[..., center_h - radius_h:center_h + radius_h, center_w - radius_w:center_w + radius_w] = 1

            # 提取低频部分，通过频率域特征图与低频掩码相乘得到
            low_freq_component = fft_x_shifted * low_freq_mask

            # 将低频部分的频率域特征图进行逆移位操作，恢复到原始的频率域布局
            low_freq_back = torch.fft.ifftshift(low_freq_component)
            # 对逆移位后的低频特征图进行二维逆傅里叶变换，转换回空间域，并取实部
            low_freq_img = torch.fft.ifft2(low_freq_back).real

            return low_freq_img

        elif return_type == 'high':
            # 构造高频掩码，初始化为全一张量
            high_freq_mask = torch.ones_like(fft_x)
            # 将中心区域（低频部分）设置为 0，其余部分为 1，对应高频部分
            high_freq_mask[..., center_h - radius_h:center_h + radius_h, center_w - radius_w:center_w + radius_w] = 0

            # 提取高频部分，通过频率域特征图与高频掩码相乘得到
            high_freq_component = fft_x_shifted * high_freq_mask

            # 将高频部分的频率域特征图进行逆移位操作，恢复到原始的频率域布局
            high_freq_back = torch.fft.ifftshift(high_freq_component)
            # 对逆移位后的高频特征图进行二维逆傅里叶变换，转换回空间域，并取实部
            high_freq_img = torch.fft.ifft2(high_freq_back).real

            return high_freq_img

        else:  # 默认返回高频和低频
            # 构造低频掩码，初始化为全零张量
            low_freq_mask = torch.zeros_like(fft_x)
            # 将中心区域设置为 1，该区域对应低频部分
            low_freq_mask[..., center_h - radius_h:center_h + radius_h, center_w - radius_w:center_w + radius_w] = 1

            # 构造高频掩码，通过 1 减去低频掩码得到
            high_freq_mask = 1 - low_freq_mask

            # 提取低频部分，通过频率域特征图与低频掩码相乘得到
            low_freq_component = fft_x_shifted * low_freq_mask
            # 将低频部分的频率域特征图进行逆移位操作，恢复到原始的频率域布局
            low_freq_back = torch.fft.ifftshift(low_freq_component)
            # 对逆移位后的低频特征图进行二维逆傅里叶变换，转换回空间域，并取实部
            low_freq_img = torch.fft.ifft2(low_freq_back).real

            # 提取高频部分，通过频率域特征图与高频掩码相乘得到
            high_freq_component = fft_x_shifted * high_freq_mask
            # 将高频部分的频率域特征图进行逆移位操作，恢复到原始的频率域布局
            high_freq_back = torch.fft.ifftshift(high_freq_component)
            # 对逆移位后的高频特征图进行二维逆傅里叶变换，转换回空间域，并取实部
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
    """
    定义 ResNet 网络类，支持光学骨干网络模式下结合 SAR 图像进行特征处理。

    Args:
        block (nn.Module): 残差块类型，如 BasicBlock 或 Bottleneck。
        layers (list): 每个阶段残差块的数量列表。
        num_classes (int, 可选): 分类任务的类别数，默认为 1000。
        opt_backbone (bool, 可选): 是否启用光学骨干网络模式，默认为 False。
    """
    def __init__(self, block, layers, num_classes=1000, opt_backbone=False):
        # 调用父类的构造函数进行初始化
        super(ResNet, self).__init__()
        # 初始输入通道数
        self.inplanes = 128
        # 第一个 3x3 卷积层，输入通道为 3，输出通道为 64，步长为 2
        self.conv1 = conv3x3(3, 64, stride=2)
        # 第一个批量归一化层
        self.bn1 = nn.BatchNorm2d(64)
        # 第一个 ReLU 激活函数
        self.relu1 = nn.ReLU(inplace=True)
        # 第二个 3x3 卷积层，输入输出通道均为 64
        self.conv2 = conv3x3(64, 64)
        # 第二个批量归一化层
        self.bn2 = nn.BatchNorm2d(64)
        # 第二个 ReLU 激活函数
        self.relu2 = nn.ReLU(inplace=True)
        # 第三个 3x3 卷积层，输入通道为 64，输出通道为 128
        self.conv3 = conv3x3(64, 128)
        # 第三个批量归一化层
        self.bn3 = nn.BatchNorm2d(128)
        # 第三个 ReLU 激活函数
        self.relu3 = nn.ReLU(inplace=True)
        # 最大池化层
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 构建第一个残差层
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 构建第二个残差层，步长为 2
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 构建第三个残差层，步长为 2
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # 构建第四个残差层，步长为 2
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # 是否启用光学骨干网络模式
        self.opt_backbone = opt_backbone
        # 频率分离模块
        self.FrequencySeparation = FrequencySeparation()
        # 平均池化层（注释掉，未使用）
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # 全连接层（注释掉，未使用）
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化网络权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 计算卷积层的参数数量
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # 使用正态分布初始化卷积层的权重
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                # 初始化批量归一化层的权重为 1
                m.weight.data.fill_(1)
                # 初始化批量归一化层的偏置为 0
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        构建一个残差层，包含多个残差块。

        Args:
            block (nn.Module): 残差块类型。
            planes (int): 残差块的输出通道数。
            blocks (int): 残差块的数量。
            stride (int, 可选): 第一个残差块的步长，默认为 1。

        Returns:
            nn.Sequential: 由多个残差块组成的残差层。
        """
        downsample = None
        # 当步长不为 1 或者输入输出通道数不匹配时，需要下采样
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
        # 添加第一个残差块，可能包含下采样
        layers.append(block(self.inplanes, planes, stride, downsample))
        # 更新输入通道数
        self.inplanes = planes * block.expansion
        # 添加剩余的残差块
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, x_sar=None):
        """
        定义 ResNet 网络的前向传播过程。

        Args:
            x (torch.Tensor): 输入的光学图像张量。
            x_sar (torch.Tensor, 可选): 输入的 SAR 图像张量，默认为 None。

        Returns:
            list: 包含不同阶段特征图的列表。
        """
        # 保留与原代码相同的结构，确保 API 兼容性
        # 经过第一个卷积层、批量归一化层和激活函数
        x = self.relu1(self.bn1(self.conv1(x)))
        # 经过第二个卷积层、批量归一化层和激活函数
        x = self.relu2(self.bn2(self.conv2(x)))
        # 经过第三个卷积层、批量归一化层和激活函数
        x = self.relu3(self.bn3(self.conv3(x)))
        
        # 保存初始特征
        feat = []
        feat.append(x)
        
        # 应用最大池化
        x = self.maxpool(x)
        
        # 通过 4 个残差块，保留 GPU 优化
        x = self.layer1(x)
        feat.append(x)
        x = self.layer2(x)
        feat.append(x)
        x = self.layer3(x)
        feat.append(x)
        x = self.layer4(x)
        feat.append(x)
        
        # 如果是光学骨干网络模式且提供了 SAR 输入
        if self.opt_backbone and x_sar is not None:
            # 对 SAR 图像执行相同的特征提取，保持与原始代码相同的调用方式
            x_sar = self.relu1(self.bn1(self.conv1(x_sar)))
            x_sar = self.relu2(self.bn2(self.conv2(x_sar)))
            x_sar = self.relu3(self.bn3(self.conv3(x_sar)))
            
            # 保存 SAR 特征
            feat_sar = []
            feat_sar.append(x_sar)
            
            # 应用最大池化
            x_sar = self.maxpool(x_sar)
            
            # 自适应实例归一化（AdaIN）用于特征对齐
            # 层 1 的处理
            x_sar = self.layer1(x_sar)
            x_sar = adain(x_sar, feat[1])
            feat_sar.append(x_sar)
            
            # 层 2 的处理
            x_sar = self.layer2(x_sar)
            x_sar = adain(x_sar, feat[2])
            feat_sar.append(x_sar)
            
            # 层 3 和层 4 的处理（不使用 AdaIN）
            x_sar = self.layer3(x_sar)
            feat_sar.append(x_sar)
            x_sar = self.layer4(x_sar)
            feat_sar.append(x_sar)
            
            # 实现频率域分离（可选、高级特性）
            if hasattr(self, 'freq_sep') and self.freq_sep is not None:
                # 获取高频和低频分量
                for i in range(len(feat)):
                    # 高频分量保持不变，只处理低频分量
                    feat_h = self.FrequencySeparation(feat[i], return_type='high')
                    feat_l_sar = self.FrequencySeparation(feat_sar[i], return_type='low')
                    
                    # 组合高频光学特征和低频 SAR 特征
                    feat[i] = feat_h + feat_l_sar
            
            return feat
        
        # 如果没有 SAR 输入或不是光学骨干网络模式，直接返回特征列表
        return feat

        # 下面代码被注释掉，未使用
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        # return x


def resnet18(pretrained=False, **kwargs):
    """
    构建一个 ResNet-18 模型。

    Args:
        pretrained (bool): 如果为 True，则返回一个在 Places 数据集上预训练的模型。
        **kwargs: 传递给 ResNet 类构造函数的其他关键字参数。

    Returns:
        nn.Module: 构建好的 ResNet-18 模型。
    """
    # 实例化 ResNet 类，使用 BasicBlock 作为残差块，每个阶段的残差块数量为 [2, 2, 2, 2]
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    # 如果 pretrained 为 True，则加载预训练权重
    if pretrained:
        # 从指定 URL 加载预训练权重并加载到模型中，strict=False 表示不严格匹配模型参数
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


class LightweightResNet(nn.Module):
    """轻量化版的ResNet，通道数减少但结构简单可靠"""
    def __init__(self, block, layers, channel_reduction=0.5, num_classes=1000):
        super(LightweightResNet, self).__init__()
        self.channel_reduction = channel_reduction
        # 如果channel_reduction=0.5，则scale=0.5
        self.scale = 1.0 - channel_reduction
        
        # 计算各层通道数
        self.channels = {
            'conv1': int(64 * self.scale),
            'conv2': int(64 * self.scale),
            'conv3': int(128 * self.scale),
            'layer1': int(64 * self.scale),
            'layer2': int(128 * self.scale),
            'layer3': int(256 * self.scale),
            'layer4': int(512 * self.scale)
        }
        
        # 定义网络层
        self.inplanes = self.channels['conv3']
        self.conv1 = conv3x3(3, self.channels['conv1'], stride=2)
        self.bn1 = nn.BatchNorm2d(self.channels['conv1'])
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(self.channels['conv1'], self.channels['conv2'])
        self.bn2 = nn.BatchNorm2d(self.channels['conv2'])
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(self.channels['conv2'], self.channels['conv3'])
        self.bn3 = nn.BatchNorm2d(self.channels['conv3'])
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 四个阶段的残差块
        self.layer1 = self._make_layer(block, self.channels['layer1'], layers[0])
        self.layer2 = self._make_layer(block, self.channels['layer2'], layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.channels['layer3'], layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.channels['layer4'], layers[3], stride=2)
        
        # 初始化所有层的权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        """创建残差层"""
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

    def forward(self, x, x_sar=None):
        """前向传播函数"""
        # 初始处理
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        
        # 保存特征
        feat = []
        feat.append(x)
        
        # 应用最大池化
        x = self.maxpool(x)
        
        # 通过残差块
        x = self.layer1(x)
        feat.append(x)
        x = self.layer2(x)
        feat.append(x)
        x = self.layer3(x)
        feat.append(x)
        x = self.layer4(x)
        feat.append(x)
        
        return feat


def lightweight_resnet18(pretrained=False, channel_reduction=0.5, **kwargs):
    """构建轻量化的ResNet-18模型。

    参数:
        pretrained (bool): 不使用，保留参数保持API兼容性
        channel_reduction (float): 通道数减少比例，默认为0.5（减少50%）
    """
    print(f"初始化轻量化ResNet18模型（通道减少{channel_reduction*100:.0f}%），使用随机初始化权重")
    
    # 创建轻量化模型，不加载预训练权重
    model = LightweightResNet(
        BasicBlock, 
        [2, 2, 2, 2], 
        channel_reduction=channel_reduction, 
        **kwargs
    )
    
    return model


if __name__ == '__main__':
    model = resnet18(pretrained=True)
    data = torch.randn(1, 3, 512, 512)
    output = model(data)
    print(output[0].shape, output[1].shape, output[2].shape, output[3].shape, output[4].shape)
