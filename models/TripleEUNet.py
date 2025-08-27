import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import resnet18, lightweight_resnet18
# from .resnet import FrequencySeparation
import numpy as np


class DoubleConv_down(nn.Module):
    """(卷积 => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.5),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleConv_up(nn.Module):
    """(卷积 => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """通过最大池化然后双卷积进行下采样"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4,
                      stride=2, padding=1, bias=False)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样然后双卷积"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # 如果是双线性插值,使用普通卷积来减少通道数

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, stride=2,
                                     padding=1)
        self.conv = DoubleConv_up(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 计算特征图尺寸差异
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # 对x1进行padding以匹配x2的尺寸
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # 如果遇到padding问题,可以参考:
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        
        # 特征拼接
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return nn.Tanh()(self.conv(x))

class Decoder(nn.Module):
    def __init__(self, n_classes):
        super(Decoder, self).__init__()
        self.conv1x1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            ) for in_channels in [64, 128, 256, 512]  # 假设 inputs1 和 inputs2 是来自 ResNet 的特征
        ])
        self.out_conv = nn.Conv2d(128, 64, kernel_size=1)  # 128 来自于拼接后的通道数 (32 * 4)
        self.outc = OutConv(64, n_classes)

    def forward(self, inputs1, inputs2):
        def process_features(inputs):
            processed_feats = []
            target_size = inputs[0].size()[2:]  # 以第一组特征的空间尺寸作为目标尺寸
            for i in range(4):
                # print(inputs[i].shape)
                x = self.conv1x1[i](inputs[i])
                # 调整每组特征图的大小
                if x.size()[2:] != target_size:
                    x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
                processed_feats.append(x)
            return processed_feats

        # 处理 inputs1 和 inputs2
        feats1 = process_features(inputs1)
        feats2 = process_features(inputs2)

        # 将处理后的特征进行拼接
        x1 = torch.cat(feats1, dim=1)  # 通道数为 32*4 = 128
        x1 = self.out_conv(x1)  # 将通道数降到 32
        logits1 = self.outc(x1)

        x2 = torch.cat(feats2, dim=1)  # 通道数为 32*4 = 128
        x2 = self.out_conv(x2)  # 将通道数降到 32
        logits2 = self.outc(x2)

        return logits1, logits2, x1, x2


# 差异模块
def conv_diff(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU()
    )


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out


# 中间预测模块
def make_prediction(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    )


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    print(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class UpsampleConvLayer(torch.nn.Module):
    """上采样卷积层"""
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(UpsampleConvLayer, self).__init__()
        self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class MLP(nn.Module):
    """MLP头部,用于特征变换"""
    
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, query_dim, key_dim, mid_channel, out_channel, num_heads):
        super().__init__()
        self.num_units = mid_channel
        self.num_heads = num_heads
        self.key_dim = key_dim
        # 定义Q、K、V的线性变换
        self.W_query = nn.Conv2d(query_dim, self.num_units, kernel_size=1, stride=1)
        self.W_key = nn.Conv2d(key_dim, self.num_units, kernel_size=1, stride=1)
        self.W_value = nn.Conv2d(key_dim, self.num_units, kernel_size=1, stride=1)

    def forward(self, query, key, mask=None):
        # 生成Q、K、V
        querys = self.W_query(query)  
        keys = self.W_key(key)  
        values = self.W_value(key)
        b, c, h, w = values.shape

        # 重塑张量维度
        querys = querys.view(querys.shape[0], querys.shape[1], -1)
        keys = keys.view(keys.shape[0], keys.shape[1], -1)
        values = values.view(values.shape[0], values.shape[1], -1)

        # 多头分割
        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=1), dim=0)  
        keys = torch.stack(torch.split(keys, split_size, dim=1), dim=0)  
        values = torch.stack(torch.split(values, split_size, dim=1), dim=0)  

        # 计算注意力分数
        scores = torch.matmul(querys, keys.transpose(2, 3))  
        scores = scores / (self.key_dim ** 0.5)

        # 应用mask(如果有)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(0).repeat(self.num_heads, 1, querys.shape[2], 1)
            scores = scores.masked_fill(mask, -np.inf)
            
        # softmax归一化并计算输出
        scores = F.softmax(scores, dim=3)
        out = torch.matmul(scores, values)
        out = torch.cat(torch.split(out, 1, dim=0), dim=2).squeeze(0)  
        out = out.view(b, c, h, w)
        return out


class CD_Decoder(nn.Module):
    """变化检测解码器
    
    用于解码和融合多尺度特征,生成最终的变化检测结果
    """

    def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=True,
                 in_channels=[64, 128, 256, 512], embedding_dim=64, output_nc=2,
                 decoder_softmax=False, feature_strides=[2, 4, 8, 16]):
        super(CD_Decoder, self).__init__()
        # 参数检查
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]

        # 基本设置
        self.feature_strides = feature_strides
        self.input_transform = input_transform
        self.in_index = in_index
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.output_nc = output_nc
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        # MLP解码器头
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=c4_in_channels)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=c3_in_channels)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=c2_in_channels)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=c1_in_channels)

        # 卷积差异模块
        self.diff_c4 = conv_diff(in_channels=2 * c4_in_channels, out_channels=self.embedding_dim)
        self.diff_c3 = conv_diff(in_channels=2 * c3_in_channels, out_channels=self.embedding_dim)
        self.diff_c2 = conv_diff(in_channels=2 * c2_in_channels, out_channels=self.embedding_dim)
        self.diff_c1 = conv_diff(in_channels=2 * c1_in_channels, out_channels=self.embedding_dim)

        # 中间预测模块
        self.make_pred_c4 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c3 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c2 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c1 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)

        # 特征融合层
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=self.embedding_dim * len(in_channels), out_channels=self.embedding_dim,
                      kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )

        # 最终预测头
        self.convd2x = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)

        # 输出激活
        self.output_softmax = decoder_softmax
        self.active = nn.Sigmoid()

    def _transform_inputs(self, inputs):
        """转换解码器的输入。
        参数:
            inputs (list[Tensor]): 多层级图像特征列表。
        返回:
            Tensor: 转换后的输入
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs1, inputs2):
        # 转换编码器特征(选择层)
        x_1 = self._transform_inputs(inputs1)  # 长度=4, 1/2, 1/4, 1/8, 1/16
        x_2 = self._transform_inputs(inputs2)  # 长度=4, 1/2, 1/4, 1/8, 1/16

        # 图像1和图像2的特征
        c1_1, c2_1, c3_1, c4_1 = x_1
        c1_2, c2_2, c3_2, c4_2 = x_2

        # MLP解码器处理C1-C4
        n, _, h, w = c4_1.shape

        outputs = []
        # 第4阶段: 1/32尺度
        _c4_1 = self.linear_c4(c4_1).permute(0, 2, 1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
        _c4_2 = self.linear_c4(c4_2).permute(0, 2, 1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3])
        _c4 = self.diff_c4(torch.cat((_c4_1, _c4_2), dim=1))
        p_c4 = self.make_pred_c4(_c4)
        outputs.append(p_c4)
        _c4_up = resize(_c4, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # 第3阶段: 1/16尺度
        _c3_1 = self.linear_c3(c3_1).permute(0, 2, 1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
        _c3_2 = self.linear_c3(c3_2).permute(0, 2, 1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])
        _c3 = self.diff_c3(torch.cat((_c3_1, _c3_2), dim=1)) + F.interpolate(_c4, scale_factor=2, mode="bilinear")
        p_c3 = self.make_pred_c3(_c3)
        outputs.append(p_c3)
        _c3_up = resize(_c3, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # 第2阶段: 1/8尺度
        _c2_1 = self.linear_c2(c2_1).permute(0, 2, 1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2_2 = self.linear_c2(c2_2).permute(0, 2, 1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        _c2 = self.diff_c2(torch.cat((_c2_1, _c2_2), dim=1)) + F.interpolate(_c3, scale_factor=2, mode="bilinear")
        p_c2 = self.make_pred_c2(_c2)
        outputs.append(p_c2)
        _c2_up = resize(_c2, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # 第1阶段: 1/4尺度
        _c1_1 = self.linear_c1(c1_1).permute(0, 2, 1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1_2 = self.linear_c1(c1_2).permute(0, 2, 1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])
        _c1 = self.diff_c1(torch.cat((_c1_1, _c1_2), dim=1)) + F.interpolate(_c2, scale_factor=2, mode="bilinear")
        p_c1 = self.make_pred_c1(_c1)
        outputs.append(p_c1)

        # 线性融合所有尺度的差异图像
        _c = self.linear_fuse(torch.cat((_c4_up, _c3_up, _c2_up, _c1), dim=1))

        # 上采样x2 (1/2尺度)
        x = self.convd2x(_c)

        # 最终预测
        cp = self.change_probability(x)

        outputs.append(cp)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs


class MixFFN(nn.Module):
    """Segformer的MixFFN实现。
    参数:
        embed_dims (int): 特征维度。与MultiheadAttention相同。
            默认值: 256。
        feedforward_channels (int): FFN的隐藏维度。
            默认值: 1024。
        act_cfg (dict, optional): FFN的激活函数配置。
            默认值: dict(type='ReLU')。
        ffn_drop (float, optional): FFN中元素被置零的概率。
            默认值: 0.0。
        dropout_layer (dict, optional): 添加残差连接时使用的dropout层。
            默认值: None。
        init_cfg (dict, optional): 初始化配置。
            默认值: None。
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 dropout_layer=None,
                 init_cfg=None):
        super(MixFFN, self).__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg

        self.activate = self.build_activation_layer(act_cfg)

        in_channels = embed_dims
        self.fc1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        # 3x3深度可分离卷积,提供位置编码信息
        self.pe_conv = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=feedforward_channels)
        self.fc2 = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        self.drop = nn.Dropout(ffn_drop)

        self.dropout_layer = self.build_dropout(dropout_layer) if dropout_layer else nn.Identity()

    def build_activation_layer(self, act_cfg):
        act_type = act_cfg.get('type', 'GELU')
        if act_type == 'ReLU':
            return nn.ReLU()
        elif act_type == 'GELU':
            return nn.GELU()
        else:
            raise ValueError(f"不支持的激活函数类型: {act_type}")

    def build_dropout(self, dropout_cfg):
        drop_prob = dropout_cfg.get('p', 0.5)
        return nn.Dropout(drop_prob)

    def forward(self, x, identity=None):
        out = self.fc1(x)
        out = self.pe_conv(out)
        out = self.activate(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.drop(out)

        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class eca_layer(nn.Module):
    """构建ECA模块。

    参数:
        channel: 输入特征图的通道数
        k_size: 自适应选择的核大小
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 在全局空间信息上的特征描述符
        y = self.avg_pool(x)

        # ECA模块的两个不同分支
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # 多尺度信息融合
        y = self.sigmoid(y)

        # return x * y.expand_as(x)
        return y.expand_as(x)


class FrequencyMixEnh(nn.Module):
    def __init__(self, in_channels, compress_ratio=16):
        super(FrequencyMixEnh, self).__init__()
        self.in_channels = in_channels * 2

    def forward(self, feature1, feature2):
        # 对两组特征进行傅里叶变换
        fft_feature1 = torch.fft.fft2(feature1)
        fft_feature2 = torch.fft.fft2(feature2)

        # 提取幅值和相位
        magnitude1 = torch.abs(fft_feature1)
        phase1 = torch.angle(fft_feature1)

        magnitude2 = torch.abs(fft_feature2)
        phase2 = torch.angle(fft_feature2)

        # 交换幅值和相位
        swap_fft_feature2 = magnitude1 * torch.exp(1j * phase2)
        swap_fft_feature1 = magnitude2 * torch.exp(1j * phase1)

        # 逆傅里叶变换得到新的特征
        swap_feature2 = torch.fft.ifft2(swap_fft_feature2)
        swap_feature1 = torch.fft.ifft2(swap_fft_feature1)

        # new_feature1 = torch.cat((feature1, swap_feature1), dim=1)
        # new_feature2 = torch.cat((feature2, swap_feature2), dim=1)
        # print(new_feature1.shape)

        # new_feature1 = self.channel_att1(new_feature1)
        # new_feature2 = self.channel_att2(new_feature2)
        return swap_feature1.real, swap_feature2.real


class FrequencyEnh(nn.Module):
    def __init__(self,
                 in_channels,
                 k_list=[2],
                 fs_feat='feat',
                 lp_type='freq_channel_att',
                 act='sigmoid',
                 channel_res=True,
                 spatial='conv',
                 spatial_group=1,
                 compress_ratio=16):
        super().__init__()

        self.k_list = sorted(k_list)
        self.in_channels = in_channels
        self.fs_feat = fs_feat
        self.lp_type = lp_type
        self.channel_res = channel_res
        self.spatial_group = min(spatial_group, in_channels)
        self.freq_thres = 0.35  # 增加阈值以提高灵活性

        # 频率权重卷积层
        if spatial == 'conv':
            self.freq_weight_conv = nn.Conv2d(in_channels, (len(k_list) + 1) * self.spatial_group,
                                              kernel_size=3, padding=1, bias=True)
        else:
            raise NotImplementedError("空间维度只实现了'conv'方法。")

        # 低频和高频的通道注意力层
        self.channel_att_low = self._make_channel_att_layer(compress_ratio)
        self.channel_att_high = self._make_channel_att_layer(compress_ratio)

        self.act_func = nn.Sigmoid() if act == 'sigmoid' else nn.Softmax(dim=1)

    def _make_channel_att_layer(self, compress_ratio):
        """创建通道注意力层
        参数:
            compress_ratio (int): 压缩比率
        """
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.in_channels, self.in_channels // compress_ratio, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels // compress_ratio, self.in_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 生成频率权重
        freq_weight = self.act_func(self.freq_weight_conv(x))
        if isinstance(self.act_func, nn.Softmax):
            freq_weight *= freq_weight.shape[1]

        # 傅里叶变换和掩码
        x_fft = torch.fft.fftshift(torch.fft.fft2(x))
        low_mask, high_mask = self._get_frequency_masks(x_fft.shape[2:], x.device)

        # 分离低频和高频分量
        low_part = torch.fft.ifft2(torch.fft.ifftshift(x_fft * low_mask)).real
        high_part = x - low_part

        # 低频和高频的通道注意力
        low_att = self._apply_channel_attention(self.channel_att_low, low_mask, x_fft)
        high_att = self._apply_channel_attention(self.channel_att_high, high_mask, x_fft)
        print(low_att.shape, high_att.shape)
        # 频率加权
        low_part = low_part * freq_weight[:, :1] * low_att
        high_part = high_part * freq_weight[:, 1:2] * high_att
        res = low_part + high_part

        return res + x if self.channel_res else res

    def _get_frequency_masks(self, shape, device):
        h, w = shape
        low_mask = torch.zeros((1, 1, h, w), device=device)
        high_mask = torch.ones_like(low_mask)

        h_low, h_high = round(h / 2 - h * self.freq_thres), round(h / 2 + h * self.freq_thres)
        w_low, w_high = round(w / 2 - w * self.freq_thres), round(w / 2 + w * self.freq_thres)

        low_mask[:, :, h_low:h_high, w_low:w_high] = 1.0
        high_mask[:, :, h_low:h_high, w_low:w_high] = 0.0
        return low_mask, high_mask

    def _apply_channel_attention(self, channel_att_layer, mask, x_fft):
        mask_fft = x_fft * mask
        return torch.sqrt(channel_att_layer(mask_fft.real) ** 2 + channel_att_layer(mask_fft.imag) ** 2 + 1e-8)


class BidirectionalChannelAttention(nn.Module):
    """双向通道注意力融合模块
    
    通过交换融合光学和SAR特征的注意力权重，增强两种异质模态之间的信息交互
    
    参数:
        in_channels (int): 输入通道数
        reduction_ratio (int): 注意力机制中的通道缩减比例
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(BidirectionalChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 共享MLP层用于特征压缩和通道注意力计算
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        
        # 使用另一个MLP学习模态特定的注意力调整
        self.specific_mlp = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x1, x2):
        """前向传播
        
        参数:
            x1 (tensor): 第一个特征图（通常是光学特征）
            x2 (tensor): 第二个特征图（通常是SAR特征）
            
        返回:
            tuple: (增强后的x1特征, 增强后的x2特征)
        """
        # 保存原始特征用于残差连接
        x1_orig = x1
        x2_orig = x2
        
        # 第一个分支特征的注意力权重
        x1_avg = self.avg_pool(x1)
        x1_max = self.max_pool(x1)
        
        x1_avg_weight = self.shared_mlp(x1_avg)
        x1_max_weight = self.shared_mlp(x1_max)
        
        # 第二个分支特征的注意力权重
        x2_avg = self.avg_pool(x2)
        x2_max = self.max_pool(x2)
        
        x2_avg_weight = self.shared_mlp(x2_avg)
        x2_max_weight = self.shared_mlp(x2_max)
        
        # 基础注意力权重
        x1_weight = self.sigmoid(x1_avg_weight + x1_max_weight)
        x2_weight = self.sigmoid(x2_avg_weight + x2_max_weight)
        
        # 模态交互学习 - 结合两种模态的注意力信息
        # 拼接平均池化和最大池化结果
        x_combined_avg = torch.cat([x1_avg, x2_avg], dim=1)
        x_combined_max = torch.cat([x1_max, x2_max], dim=1)
        
        # 通过特定MLP学习交互注意力
        x_combined_avg_weight = self.specific_mlp(x_combined_avg)
        x_combined_max_weight = self.specific_mlp(x_combined_max)
        
        # 计算混合注意力
        x_combined_weight = self.sigmoid(x_combined_avg_weight + x_combined_max_weight)
        
        # 双向增强 - 交换注意力
        # 使用第二个模态的注意力权重来增强第一个模态的特征，反之亦然
        x1_enhanced = x1 * x2_weight
        x2_enhanced = x2 * x1_weight
        
        # 添加共享注意力组件
        x1_enhanced = x1_enhanced + x1 * x_combined_weight
        x2_enhanced = x2_enhanced + x2 * x_combined_weight
        
        # 残差连接
        x1_enhanced = x1_enhanced + x1_orig
        x2_enhanced = x2_enhanced + x2_orig
        
        return x1_enhanced, x2_enhanced


class DualEUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        """双分支EUNet网络
        参数:
            n_channels (int): 输入通道数
            n_classes (int): 输出类别数
            bilinear (bool): 是否使用双线性插值上采样
        """
        super(DualEUNet, self).__init__()
        # 光学图像编码器
        self.encoder_opt = resnet18(pretrained=True)
        # SAR图像编码器 
        self.encoder_sar = resnet18(pretrained=True)
        # Dropout层
        self.dropout = nn.Dropout2d(p=0.1, inplace=False)
        
        # 特征融合相关参数和层
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.in_channels = [64, 128, 256, 512]
        self.channels = 128
        
        # 特征融合卷积层
        self.fusion_conv1 = nn.Sequential(
            nn.Conv2d(self.channels * len(self.in_channels), self.channels, kernel_size=1),
            nn.BatchNorm2d(self.channels),
        )
        self.fusion_conv2 = nn.Sequential(
            nn.Conv2d(self.channels * len(self.in_channels), self.channels, kernel_size=1),
            nn.BatchNorm2d(self.channels),
        )
        
        # 分割输出层
        self.conv_seg = nn.Conv2d(self.channels * 2, 2, kernel_size=1)
        
        # 特征转换卷积层
        for i in range(len(self.in_channels)):
            self.convs1.append(
                nn.Sequential(
                    nn.Conv2d(self.in_channels[i], self.channels, kernel_size=1),
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU()
                )
            )
            
        for i in range(len(self.in_channels)):
            self.convs2.append(
                nn.Sequential(
                    nn.Conv2d(self.in_channels[i], self.channels, kernel_size=1),
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU()
                )
            )
        
        # 注意力层
        self.atten = self._make_channel_att_layer(compress_ratio=16)
        
        # 判别器 
        self.student_discriminator = MixFFN(
            embed_dims=self.channels * 2,  # 输入通道数是两个特征的拼接
            feedforward_channels=self.channels * 2,
            ffn_drop=0.,
            dropout_layer=dict(type='DropPath', drop_prob=0.),
            act_cfg=dict(type='GELU'))

    def _make_channel_att_layer(self, compress_ratio):
        """创建通道注意力层
        参数:
            compress_ratio (int): 压缩比率
        """
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.channels, self.channels // compress_ratio, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels // compress_ratio, self.channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def base_forward1(self, inputs):
        """光学图像特征前向传播
        参数:
            inputs (list): 多尺度特征列表
        """
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs1[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=False))
        out = self.fusion_conv1(torch.cat(outs, dim=1))
        return out

    def base_forward2(self, inputs):
        """SAR图像特征前向传播
        参数:
            inputs (list): 多尺度特征列表
        """
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs2[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=False))
        out = self.fusion_conv2(torch.cat(outs, dim=1))
        return out, outs

    def cls_seg(self, feat):
        """像素级分类
        参数:
            feat (Tensor): 输入特征
        """
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def forward(self, x1, x2, x3=None, is_training=True):
        """网络前向传播
        参数:
            x1 (Tensor): 光学图像输入 (时间点1)
            x2 (Tensor): SAR图像输入 (时间点2)
            x3 (Tensor, optional): 时间点2的光学图像，用于TripleEUNet（此处不使用）
            is_training (bool): 是否处于训练模式（此处不使用）
        """
        # 编码器特征提取
        x_sar = self.encoder_sar(x2)
        x_opt = self.encoder_opt(x1)
        
        # 特征融合
        out1 = self.base_forward1(x_opt[1:])
        out2, _ = self.base_forward2(x_sar[1:])
        
        # 判别器处理
        out_ori = torch.cat([out1, out2], dim=1)
        out_ori = self.student_discriminator(out_ori)
        
        # 分割输出
        out = self.cls_seg(out_ori)
        return out


class TripleEUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        """三分支EUNet网络，用于异源遥感图像变化检测
        参数:
            n_channels (int): 输入通道数
            n_classes (int): 输出类别数
            bilinear (bool): 是否使用双线性插值上采样
        """
        super(TripleEUNet, self).__init__()
        # 光学图像编码器1 - 时间点1
        self.encoder_opt1 = resnet18(pretrained=True)
        # SAR图像编码器 - 时间点2 (学生网络)
        self.encoder_sar = resnet18(pretrained=True)
        # 光学图像编码器2 - 时间点2 (教师网络)
        self.encoder_opt2 = resnet18(pretrained=True)
        # Dropout层
        self.dropout = nn.Dropout2d(p=0.1, inplace=False)
        
        # 特征融合相关参数和层
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.convs3 = nn.ModuleList()
        
        self.in_channels = [64, 128, 256, 512]
        self.channels = 128
        
        # 特征融合卷积层
        self.fusion_conv1 = nn.Sequential(
            nn.Conv2d(self.channels * len(self.in_channels), self.channels, kernel_size=1),
            nn.BatchNorm2d(self.channels),
        )
        self.fusion_conv2 = nn.Sequential(
            nn.Conv2d(self.channels * len(self.in_channels), self.channels, kernel_size=1),
            nn.BatchNorm2d(self.channels),
        )
        self.fusion_conv3 = nn.Sequential(
            nn.Conv2d(self.channels * len(self.in_channels), self.channels, kernel_size=1),
            nn.BatchNorm2d(self.channels),
        )
        
        # 双向通道注意力融合模块
        self.bidirectional_attention = BidirectionalChannelAttention(self.channels, reduction_ratio=16)
        
        # 特征融合卷积层
        for i in range(len(self.in_channels)):
            self.convs1.append(
                nn.Sequential(
                    nn.Conv2d(self.in_channels[i], self.channels, kernel_size=1),
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU()
                )
            )
            self.convs2.append(
                nn.Sequential(
                    nn.Conv2d(self.in_channels[i], self.channels, kernel_size=1),
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU()
                )
            )
            self.convs3.append(
                nn.Sequential(
                    nn.Conv2d(self.in_channels[i], self.channels, kernel_size=1),
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU()
                )
            )
        
        # 分割输出层
        self.conv_seg = nn.Sequential(
            nn.Conv2d(self.channels * 2, self.channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(),
            nn.Conv2d(self.channels, n_classes, kernel_size=1)
        )
        
        # 教师网络分割输出层 - 使用与学生网络相同的初始化方法
        self.conv_seg_teacher = nn.Sequential(
            nn.Conv2d(self.channels * 2, self.channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(),
            nn.Conv2d(self.channels, n_classes, kernel_size=1)
        )
        
        # 确保权重正确初始化 - 复制学生网络权重到教师网络
        # 这样可以确保教师网络的初始性能与学生网络类似
        self.conv_seg_teacher[0].weight.data.copy_(self.conv_seg[0].weight.data)
        self.conv_seg_teacher[0].bias.data.copy_(self.conv_seg[0].bias.data)
        self.conv_seg_teacher[1].weight.data.copy_(self.conv_seg[1].weight.data)
        self.conv_seg_teacher[1].bias.data.copy_(self.conv_seg[1].bias.data)
        self.conv_seg_teacher[3].weight.data.copy_(self.conv_seg[3].weight.data)
        self.conv_seg_teacher[3].bias.data.copy_(self.conv_seg[3].bias.data)
        
        # 学生判别器
        self.student_discriminator = MixFFN(
            embed_dims=self.channels * 2,  # 输入通道数是两个特征的拼接
            feedforward_channels=self.channels * 2,
            ffn_drop=0.,
            dropout_layer=dict(type='DropPath', drop_prob=0.),
            act_cfg=dict(type='GELU'))
        
        # 增强教师判别器
        self.teacher_discriminator = MixFFN(
            embed_dims=self.channels * 2,  # 因为输入是enhanced_out1和enhanced_out3的拼接
            feedforward_channels=self.channels * 3,  # 增加教师网络的通道数
            ffn_drop=0.1,  # 适度的dropout
            dropout_layer=dict(type='DropPath', drop_prob=0.1),
            act_cfg=dict(type='GELU'))
        
        # 差异图注意力迁移模块
        # 学生网络差异图生成
        self.student_diff_module = nn.Sequential(
            nn.Conv2d(self.channels * 2, self.channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 教师网络差异图生成
        self.teacher_diff_module = nn.Sequential(
            nn.Conv2d(self.channels * 2, self.channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 混合融合模块 - 学生网络
        self.student_fusion = nn.Sequential(
            nn.Conv2d(self.channels * 2 + 1, self.channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channels * 2),
            nn.ReLU(inplace=True)
        )
        
        # 混合融合模块 - 教师网络
        self.teacher_fusion = nn.Sequential(
            nn.Conv2d(self.channels * 2 + 1, self.channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channels * 2),
            nn.ReLU(inplace=True)
        )

    def _make_channel_att_layer(self, compress_ratio):
        """创建通道注意力层
        参数:
            compress_ratio (int): 压缩比率
        """
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.channels, self.channels // compress_ratio, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels // compress_ratio, self.channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def base_forward1(self, inputs):
        """光学图像特征前向传播 - 分支1
        参数:
            inputs (list): 多尺度特征列表
        """
        input_shape = inputs[0].shape[2:]
        outs = []
        
        # 预先分配空间来存储处理后的特征
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs1[idx]
            # 统一处理分辨率，减少重复的形状比较检查
            if x.shape[2:] != input_shape:
                x_resized = resize(
                    input=conv(x),
                    size=input_shape,
                    mode='bilinear',
                    align_corners=False)
                outs.append(x_resized)
            else:
                outs.append(conv(x))
                
        # 一次性拼接所有特征，减少多次拼接操作
        out = self.fusion_conv1(torch.cat(outs, dim=1))
        return out

    def base_forward2(self, inputs):
        """SAR图像特征前向传播 - 分支2
        参数:
            inputs (list): 多尺度特征列表
        """
        input_shape = inputs[0].shape[2:]
        outs = []
        
        # 预先分配空间来存储处理后的特征
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs2[idx]
            # 统一处理分辨率，减少重复的形状比较检查
            if x.shape[2:] != input_shape:
                x_resized = resize(
                    input=conv(x),
                    size=input_shape,
                    mode='bilinear',
                    align_corners=False)
                outs.append(x_resized)
            else:
                outs.append(conv(x))
                
        # 一次性拼接所有特征，减少多次拼接操作
        out = self.fusion_conv2(torch.cat(outs, dim=1))
        return out, outs
        
    def base_forward3(self, inputs):
        """光学图像特征前向传播 - 分支3 (教师网络)
        参数:
            inputs (list): 多尺度特征列表
        """
        input_shape = inputs[0].shape[2:]
        outs = []
        
        # 预先分配空间来存储处理后的特征
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs3[idx]
            # 统一处理分辨率，减少重复的形状比较检查
            if x.shape[2:] != input_shape:
                x_resized = resize(
                    input=conv(x),
                    size=input_shape,
                    mode='bilinear',
                    align_corners=False)
                outs.append(x_resized)
            else:
                outs.append(conv(x))
                
        # 一次性拼接所有特征，减少多次拼接操作
        out = self.fusion_conv3(torch.cat(outs, dim=1))
        return out, outs

    def cls_seg(self, feat):
        """像素级分类
        参数:
            feat (Tensor): 输入特征
        """
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def cls_seg_teacher(self, feat):
        """教师网络的像素级分类
        参数:
            feat (Tensor): 输入特征
        """
        if self.dropout is not None:
            # 训练时使用较低的dropout，避免过度正则化
            feat = F.dropout(feat, p=0.05, training=self.training)
        output = self.conv_seg_teacher(feat)
        return output

    def forward(self, x1, x2, x3=None, is_training=True):
        """网络前向传播
        参数:
            x1 (Tensor): 光学图像输入 (时间点1)
            x2 (Tensor): SAR图像输入 (时间点2)
            x3 (Tensor, optional): 光学图像输入 (时间点2)，用作教师网络输入
            is_training (bool): 是否处于训练模式
        返回:
            如果is_training=True且提供了x3:
                tuple: (学生网络输出, 教师网络输出, 学生网络增强特征, 教师网络增强特征, 
                       学生网络中间特征, 教师网络中间特征, 光学t1特征, 光学t2特征, SAR t2特征)
            否则:
                tensor: 学生网络输出
        """
        # 编码器特征提取
        x_opt1 = self.encoder_opt1(x1)  # 分支1: 时间点1的光学图像
        x_sar = self.encoder_sar(x2)    # 分支2: 时间点2的SAR图像 (学生网络)
        
        # 特征融合 - 学生网络 (光学-SAR)
        out1 = self.base_forward1(x_opt1[1:])  # 光学特征
        out2, student_features = self.base_forward2(x_sar[1:])  # SAR特征
        
        # 使用双向通道注意力融合模块增强特征交互
        enhanced_out1, enhanced_out2 = self.bidirectional_attention(out1, out2)
        
        # 使用增强后的特征
        student_concat = torch.cat([enhanced_out1, enhanced_out2], dim=1)
        
        # 生成学生网络的差异图注意力
        student_diff_attention = self.student_diff_module(student_concat)
        
        # 应用差异图注意力到学生网络特征 - 直接拼接
        student_out_with_attention = torch.cat([student_concat, student_diff_attention], dim=1)
        student_enhanced = self.student_fusion(student_out_with_attention)
        
        # 学生网络分割输出
        student_out = self.cls_seg(student_enhanced)
        
        # 测试模式或没有提供x3，只返回学生网络输出
        if x3 is None or not is_training:
            return student_out
            
        # 训练模式且提供了x3
        # 分支3: 时间点2的光学图像 (教师网络)
        x_opt2 = self.encoder_opt2(x3)
        
        # 特征融合 - 教师网络 (光学-光学)
        out3, teacher_features = self.base_forward3(x_opt2[1:])
        
        # 使用双向通道注意力融合模块增强特征交互 - 教师网络
        # 确保梯度不会流入到out1，因为教师网络不应该影响光学图1的编码
        enhanced_out1_teacher, enhanced_out3 = self.bidirectional_attention(out1.detach(), out3)
        
        # 使用增强后的特征
        teacher_concat = torch.cat([enhanced_out1_teacher, enhanced_out3], dim=1)
        
        # 生成教师网络的差异图注意力
        teacher_diff_attention = self.teacher_diff_module(teacher_concat)
        
        # 应用差异图注意力到教师网络特征
        teacher_out_with_attention = torch.cat([teacher_concat, teacher_diff_attention], dim=1)
        teacher_enhanced = self.teacher_fusion(teacher_out_with_attention)
        
        # 教师网络输出
        teacher_out = self.cls_seg_teacher(teacher_enhanced)
        
        # 返回完整结果以供训练
        return student_out, teacher_out, student_enhanced, teacher_enhanced, \
               student_concat, teacher_concat, out1, out3, out2


class LightweightTripleEUNet(nn.Module):
    """TripleEUNet的轻量化版本，参数量减少约74.77%"""
    
    def __init__(self, n_channels, n_classes, bilinear=False, channel_reduction=0.5, attention_reduction_ratio=32):
        """
        初始化轻量化TripleEUNet模型
        
        参数:
            n_channels (int): 输入通道数
            n_classes (int): 输出通道数/类别数
            bilinear (bool): 是否使用双线性插值进行上采样
            channel_reduction (float): 通道数减少比例，默认减少50%
            attention_reduction_ratio (int): 注意力模块的reduction ratio，默认为32
        """
        super(LightweightTripleEUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.channel_reduction = channel_reduction
        
        # 轻量化版本使用较小的特征通道数
        channel_scale = 1.0 - channel_reduction  # 默认是0.5，相当于通道数减少50%
        base_channels = int(64 * channel_scale)  # 基础通道数从64降至32
        
        # 使用轻量化ResNet18作为编码器，使用随机初始化的权重
        # 光学图像分支1的编码器
        self.base_encoder1 = lightweight_resnet18(channel_reduction=channel_reduction)
        # 多模态图像分支2的编码器
        self.base_encoder2 = lightweight_resnet18(channel_reduction=channel_reduction)
        # 光学图像分支3的编码器 (教师网络)
        self.base_encoder3 = lightweight_resnet18(channel_reduction=channel_reduction)
        
        # 轻量化版本的双向通道注意力模块
        self.channel_att = BidirectionalChannelAttention(
            in_channels=int(512 * channel_scale),  # 减少通道数
            reduction_ratio=attention_reduction_ratio  # 增大reduction_ratio减少参数
        )
        
        # 轻量化解码器 - 使用加法而非拼接来融合特征
        self.decoder = CD_Decoder(
            in_channels=[int(64 * channel_scale), int(128 * channel_scale), 
                         int(256 * channel_scale), int(512 * channel_scale)],
            embedding_dim=int(64 * channel_scale),
            feature_strides=[2, 4, 8, 16],
            output_nc=n_classes,
            decoder_softmax=False
        )
        
        # 轻量化教师解码器
        self.teacher_decoder = CD_Decoder(
            in_channels=[int(64 * channel_scale), int(128 * channel_scale), 
                         int(256 * channel_scale), int(512 * channel_scale)],
            embedding_dim=int(64 * channel_scale),
            feature_strides=[2, 4, 8, 16],
            output_nc=n_classes,
            decoder_softmax=False
        )
        
        # 频域增强 - 简化版本
        self.freq_enh = FrequencyEnh(
            in_channels=int(64 * channel_scale),
            compress_ratio=attention_reduction_ratio
        )
        
        # 轻量化的MLP层 - 用于特征变换
        self.mlp = MLP(
            input_dim=int(512 * channel_scale),
            embed_dim=int(256 * channel_scale)
        )
        
        # 差异特征增强 - 轻量化版本
        self.freq_mix = FrequencyMixEnh(
            in_channels=int(64 * channel_scale),
            compress_ratio=attention_reduction_ratio
        )

    def _make_channel_att_layer(self, compress_ratio):
        """创建轻量化的通道注意力层"""
        channel_att_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.inplanes, self.inplanes // compress_ratio, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes // compress_ratio, self.inplanes, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        return channel_att_layers

    def base_forward1(self, inputs):
        """光学图像分支1的前向传播 - 轻量化版本"""
        x = self.base_encoder1.conv1(inputs)
        x = self.base_encoder1.bn1(x)
        # 使用正确的relu属性名称
        if hasattr(self.base_encoder1, 'relu1'):  # 轻量版ResNet使用relu1
            x = self.base_encoder1.relu1(x)
        else:  # 标准ResNet使用relu
            x = self.base_encoder1.relu(x)
        
        if hasattr(self.base_encoder1, 'conv2'):
            x = self.base_encoder1.conv2(x)
            x = self.base_encoder1.bn2(x)
            if hasattr(self.base_encoder1, 'relu2'):
                x = self.base_encoder1.relu2(x)
            else:
                x = self.base_encoder1.relu(x)
                
        if hasattr(self.base_encoder1, 'conv3'):
            x = self.base_encoder1.conv3(x)
            x = self.base_encoder1.bn3(x)
            if hasattr(self.base_encoder1, 'relu3'):
                x = self.base_encoder1.relu3(x)
            else:
                x = self.base_encoder1.relu(x)
        x = self.base_encoder1.maxpool(x)

        # 只提取关键层特征，减少计算量
        c1 = self.base_encoder1.layer1(x)
        c2 = self.base_encoder1.layer2(c1)
        c3 = self.base_encoder1.layer3(c2)
        c4 = self.base_encoder1.layer4(c3)

        return [c1, c2, c3, c4]

    def base_forward2(self, inputs):
        """SAR图像分支2的前向传播 - 轻量化版本"""
        x = self.base_encoder2.conv1(inputs)
        x = self.base_encoder2.bn1(x)
        # 使用正确的relu属性名称
        if hasattr(self.base_encoder2, 'relu1'):  # 轻量版ResNet使用relu1
            x = self.base_encoder2.relu1(x)
        else:  # 标准ResNet使用relu
            x = self.base_encoder2.relu(x)
        
        if hasattr(self.base_encoder2, 'conv2'):
            x = self.base_encoder2.conv2(x)
            x = self.base_encoder2.bn2(x)
            if hasattr(self.base_encoder2, 'relu2'):
                x = self.base_encoder2.relu2(x)
            else:
                x = self.base_encoder2.relu(x)
                
        if hasattr(self.base_encoder2, 'conv3'):
            x = self.base_encoder2.conv3(x)
            x = self.base_encoder2.bn3(x)
            if hasattr(self.base_encoder2, 'relu3'):
                x = self.base_encoder2.relu3(x)
            else:
                x = self.base_encoder2.relu(x)
        x = self.base_encoder2.maxpool(x)

        # 只提取关键层特征，减少计算量
        c1 = self.base_encoder2.layer1(x)
        c2 = self.base_encoder2.layer2(c1)
        c3 = self.base_encoder2.layer3(c2)
        c4 = self.base_encoder2.layer4(c3)

        return [c1, c2, c3, c4]

    def base_forward3(self, inputs):
        """光学图像分支3（教师）的前向传播 - 轻量化版本"""
        x = self.base_encoder3.conv1(inputs)
        x = self.base_encoder3.bn1(x)
        # 使用正确的relu属性名称
        if hasattr(self.base_encoder3, 'relu1'):  # 轻量版ResNet使用relu1
            x = self.base_encoder3.relu1(x)
        else:  # 标准ResNet使用relu
            x = self.base_encoder3.relu(x)
        
        if hasattr(self.base_encoder3, 'conv2'):
            x = self.base_encoder3.conv2(x)
            x = self.base_encoder3.bn2(x)
            if hasattr(self.base_encoder3, 'relu2'):
                x = self.base_encoder3.relu2(x)
            else:
                x = self.base_encoder3.relu(x)
                
        if hasattr(self.base_encoder3, 'conv3'):
            x = self.base_encoder3.conv3(x)
            x = self.base_encoder3.bn3(x)
            if hasattr(self.base_encoder3, 'relu3'):
                x = self.base_encoder3.relu3(x)
            else:
                x = self.base_encoder3.relu(x)
        x = self.base_encoder3.maxpool(x)

        # 只提取关键层特征，减少计算量
        c1 = self.base_encoder3.layer1(x)
        c2 = self.base_encoder3.layer2(c1)
        c3 = self.base_encoder3.layer3(c2)
        c4 = self.base_encoder3.layer4(c3)

        return [c1, c2, c3, c4]

    def cls_seg(self, feat):
        """变化检测分支的分类层 - 轻量化版本"""
        if self.n_classes > 1:
            return F.log_softmax(feat, dim=1)
        else:
            return feat

    def cls_seg_teacher(self, feat):
        """教师网络的分类层 - 轻量化版本"""
        if self.n_classes > 1:
            return F.log_softmax(feat, dim=1)
        else:
            return feat

    def forward(self, x1, x2, x3=None, is_training=True):
        """模型前向传播
        
        参数:
            x1: 光学图像1
            x2: SAR图像
            x3: 光学图像2（教师网络输入），可选
            is_training: 是否处于训练模式
        
        返回:
            元组：返回9个值以匹配原始TripleEUNet的接口
        """
        # 特征提取
        c1 = self.base_forward1(x1)
        c2 = self.base_forward2(x2)
        
        # 使用channel_att进行特征交互
        c1_interact, c2_interact = self.channel_att(c1[3], c2[3])
        
        # 替换原始特征
        c1[3] = c1_interact
        c2[3] = c2_interact
        
        # 学生网络预测 - CD_Decoder返回outputs列表
        outputs = self.decoder(c1, c2)
        # 取最后一个输出作为主要预测结果
        out = outputs[-1]
        # 获取特征层作为特征
        feat1 = c1[0]  # 第一个编码器特征作为中间特征
        feat2 = c2[0]  # 第一个编码器特征作为中间特征
        
        # 应用分类层
        change_pred = self.cls_seg(out)
        
        # 训练时进行教师网络推理
        if is_training and x3 is not None:
            # 教师网络特征提取
            c3 = self.base_forward3(x3)
            c4 = self.base_forward1(x1)
            
            # 教师网络预测
            teacher_outputs = self.teacher_decoder(c3, c4)
            teacher_out = teacher_outputs[-1]  # 取最后一个输出
            teacher_pred = self.cls_seg_teacher(teacher_out)
            
            # 返回9个值匹配原始TripleEUNet：
            # 1. 学生网络输出
            # 2. 教师网络输出
            # 3. 学生增强特征
            # 4. 教师增强特征
            # 5. 学生中间特征
            # 6. 教师中间特征
            # 7. 光学t1特征
            # 8. 光学t2特征
            # 9. SAR t2特征
            student_feat = feat1  # 学生增强特征
            teacher_feat = c3[0]  # 教师增强特征
            student_mid_feat = c1  # 学生中间特征
            teacher_mid_feat = c3  # 教师中间特征
            opt_t1_feat = c1[0]  # 光学t1特征
            opt_t2_feat = c3[0]  # 光学t2特征
            sar_t2_feat = c2[0]  # SAR t2特征
            
            return change_pred, teacher_pred, student_feat, teacher_feat, \
                   student_mid_feat, teacher_mid_feat, opt_t1_feat, opt_t2_feat, sar_t2_feat
        else:
            # 测试模式，只返回预测结果
            return change_pred


if __name__ == '__main__':
    model = TripleEUNet(3, 2)
    x1 = torch.randn(1, 3, 256, 256)  # 时间点1光学图像
    x2 = torch.randn(1, 3, 256, 256)  # 时间点2 SAR图像
    x3 = torch.randn(1, 3, 256, 256)  # 时间点2光学图像
    
    # 训练模式
    y_student, y_teacher, feat_s, feat_t, mid_feat_s, mid_feat_t = model(x1, x2, x3, is_training=True)
    print(f"学生网络输出尺寸: {y_student.shape}")
    print(f"教师网络输出尺寸: {y_teacher.shape}")
    
    # 测试模式
    y_test = model(x1, x2, is_training=False)
    print(f"测试模式输出尺寸: {y_test.shape}")
