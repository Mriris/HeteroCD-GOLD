import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import resnet18
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

        # if bilinear, use the normal convolutions to reduce the number of channels

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, stride=2,
                                     padding=1)
        self.conv = DoubleConv_up(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 输入是CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
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
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(UpsampleConvLayer, self).__init__()
        self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class MLP(nn.Module):
    """MLP头部"""
    
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, query_dim, key_dim, mid_channel, out_channel, num_heads):
        super().__init__()
        self.num_units = mid_channel
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.W_query = nn.Conv2d(query_dim, self.num_units, kernel_size=1, stride=1)
        self.W_key = nn.Conv2d(key_dim, self.num_units, kernel_size=1, stride=1)
        self.W_value = nn.Conv2d(key_dim, self.num_units, kernel_size=1, stride=1)
        # self.out_conv = nn.Conv2d(self.num_units, out_channel,kernel_size=1, stride=1)

    def forward(self, query, key, mask=None):
        querys = self.W_query(query)  # [N, T_q, num_units]

        keys = self.W_key(key)  # [N, T_k, num_units]
        # print(keys.shape)
        values = self.W_value(key)
        b, c, h, w = values.shape

        querys = querys.view(querys.shape[0], querys.shape[1], -1)
        keys = keys.view(keys.shape[0], keys.shape[1], -1)
        values = values.view(values.shape[0], values.shape[1], -1)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=1), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=1), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=1), dim=0)  # [h, N, T_k, num_units/h]
        ## score = softmax(QK^T / (d_k ** 0.5))

        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]

        scores = scores / (self.key_dim ** 0.5)
        ## mask
        if mask is not None:
            ## mask:  [N, T_k] --> [h, N, T_q, T_k]
            mask = mask.unsqueeze(1).unsqueeze(0).repeat(self.num_heads, 1, querys.shape[2], 1)
            scores = scores.masked_fill(mask, -np.inf)
        scores = F.softmax(scores, dim=3)
        out = torch.matmul(scores, values)
        # print(out.shape)
        out = torch.cat(torch.split(out, 1, dim=0), dim=2).squeeze(0)  # [N, T_q, num_units]
        # out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(b, c, h, w)
        # print(scores.shape, out.shape)
        # out = scores*values
        return out


class CD_Decoder(nn.Module):
    """
    Transformer Decoder
    """

    def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=True,
                 in_channels=[64, 128, 256, 512], embedding_dim=64, output_nc=2,
                 decoder_softmax=False, feature_strides=[2, 4, 8, 16]):
        super(CD_Decoder, self).__init__()
        # assert
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]

        # settings
        self.feature_strides = feature_strides
        self.input_transform = input_transform
        self.in_index = in_index
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.output_nc = output_nc
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        # MLP decoder heads
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=c4_in_channels)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=c3_in_channels)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=c2_in_channels)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=c1_in_channels)

        # convolutional Difference Modules
        self.diff_c4 = conv_diff(in_channels=2 * c4_in_channels, out_channels=self.embedding_dim)
        self.diff_c3 = conv_diff(in_channels=2 * c3_in_channels, out_channels=self.embedding_dim)
        self.diff_c2 = conv_diff(in_channels=2 * c2_in_channels, out_channels=self.embedding_dim)
        self.diff_c1 = conv_diff(in_channels=2 * c1_in_channels, out_channels=self.embedding_dim)

        # taking outputs from middle of the encoder
        self.make_pred_c4 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c3 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c2 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c1 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)

        # Final linear fusion layer
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=self.embedding_dim * len(in_channels), out_channels=self.embedding_dim,
                      kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )

        # Final predction head
        self.convd2x = nn.Sequential(ResidualBlock(self.embedding_dim))
        # self.dense_2x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        # self.convd1x    = nn.Sequential( ResidualBlock(self.embedding_dim))
        # self.dense_1x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)

        # Final activation
        self.output_softmax = decoder_softmax
        self.active = nn.Sigmoid()

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
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
        # Transforming encoder features (select layers)
        x_1 = self._transform_inputs(inputs1)  # len=4, 1/2, 1/4, 1/8, 1/16
        x_2 = self._transform_inputs(inputs2)  # len=4, 1/2, 1/4, 1/8, 1/16

        # img1 and img2 features
        c1_1, c2_1, c3_1, c4_1 = x_1
        c1_2, c2_2, c3_2, c4_2 = x_2

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4_1.shape

        outputs = []
        # Stage 4: x1/32 scale
        _c4_1 = self.linear_c4(c4_1).permute(0, 2, 1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
        _c4_2 = self.linear_c4(c4_2).permute(0, 2, 1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3])
        _c4 = self.diff_c4(torch.cat((_c4_1, _c4_2), dim=1))
        p_c4 = self.make_pred_c4(_c4)
        outputs.append(p_c4)
        _c4_up = resize(_c4, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 3: x1/16 scale
        _c3_1 = self.linear_c3(c3_1).permute(0, 2, 1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
        _c3_2 = self.linear_c3(c3_2).permute(0, 2, 1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])
        _c3 = self.diff_c3(torch.cat((_c3_1, _c3_2), dim=1)) + F.interpolate(_c4, scale_factor=2, mode="bilinear")
        p_c3 = self.make_pred_c3(_c3)
        outputs.append(p_c3)
        _c3_up = resize(_c3, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 2: x1/8 scale
        _c2_1 = self.linear_c2(c2_1).permute(0, 2, 1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2_2 = self.linear_c2(c2_2).permute(0, 2, 1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        _c2 = self.diff_c2(torch.cat((_c2_1, _c2_2), dim=1)) + F.interpolate(_c3, scale_factor=2, mode="bilinear")
        p_c2 = self.make_pred_c2(_c2)
        outputs.append(p_c2)
        _c2_up = resize(_c2, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 1: x1/4 scale
        _c1_1 = self.linear_c1(c1_1).permute(0, 2, 1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1_2 = self.linear_c1(c1_2).permute(0, 2, 1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])
        _c1 = self.diff_c1(torch.cat((_c1_1, _c1_2), dim=1)) + F.interpolate(_c2, scale_factor=2, mode="bilinear")
        p_c1 = self.make_pred_c1(_c1)
        outputs.append(p_c1)

        # Linear Fusion of difference image from all scales
        _c = self.linear_fuse(torch.cat((_c4_up, _c3_up, _c2_up, _c1), dim=1))
        # #Dropout
        # if dropout_ratio > 0:
        #     self.dropout = nn.Dropout2d(dropout_ratio)
        # else:
        #     self.dropout = None

        # Upsampling x2 (x1/2 scale)
        x = self.convd2x(_c)
        # Residual block
        # x = self.dense_2x(x)
        # #Upsampling x2 (x1 scale)
        # x = self.convd1x(x)
        # #Residual block
        # x = self.dense_1x(x)

        # Final prediction
        cp = self.change_probability(x)

        outputs.append(cp)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs


class MixFFN(nn.Module):
    """An implementation of MixFFN of Segformer.
    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        dropout_layer (dict, optional): The dropout_layer used
            when adding the shortcut. Default: None.
        init_cfg (dict, optional): The Config for initialization.
            Default: None.
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
        # 3x3 depth wise conv to provide positional encode information
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
            raise ValueError(f"Unsupported activation type: {act_type}")

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


class DualEUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(DualEUNet, self).__init__()
        self.encoder_opt = resnet18()
        self.encoder_sar = resnet18()
        self.gen_decoder = Decoder(n_classes)
        self.dropout = nn.Dropout2d(p=0.1, inplace=False)
        # self.cd_decoder   = CD_Decoder(input_transform='multiple_select', in_index=[1, 2, 3, 4], align_corners=False, 
        #              embedding_dim= 256, output_nc=2, 
        #             decoder_softmax = False, feature_strides=[2, 4, 8, 16])
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.in_channels = [64, 128, 256, 512]
        self.channels = 128
        self.fusion_conv1 = nn.Sequential(
            nn.Conv2d(self.channels * len(self.in_channels), self.channels // 2, kernel_size=1),
            nn.BatchNorm2d(self.channels // 2),
        )
        self.fusion_conv2 = nn.Sequential(
            nn.Conv2d(self.channels * len(self.in_channels), self.channels // 2, kernel_size=1),
            nn.BatchNorm2d(self.channels // 2),
        )
        self.fusion_gen_conv1 = nn.Sequential(
            nn.Conv2d(self.channels, self.channels // 2, kernel_size=1),
            nn.BatchNorm2d(self.channels // 2),
        )
        self.fusion_gen_conv2 = nn.Sequential(
            nn.Conv2d(self.channels, self.channels // 2, kernel_size=1),
            nn.BatchNorm2d(self.channels // 2),
        )
        self.conv_seg = nn.Conv2d(self.channels, 2, kernel_size=1)
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
        # self.cross_atten1 = MultiHeadAttention(self.channels//2,self.channels//2,self.channels//2,self.channels//2,8)
        # self.cross_atten2 = MultiHeadAttention(self.channels//2,self.channels//2,self.channels//2,self.channels//2,8)
        self.atten = self._make_channel_att_layer(compress_ratio=16)
        self.freqmixenh = FrequencyMixEnh(in_channels=self.channels)
        self.fusion_layer = nn.Sequential(nn.Conv2d(self.channels * 2, self.channels, kernel_size=1),
                                          nn.GELU())
        self.atten_fusion_ffn = MixFFN(
            embed_dims=self.channels,
            feedforward_channels=self.channels,
            ffn_drop=0.,
            dropout_layer=dict(type='DropPath', drop_prob=0.),
            act_cfg=dict(type='GELU'))
        self.discriminator = MixFFN(
            embed_dims=self.channels,
            feedforward_channels=self.channels,
            ffn_drop=0.,
            dropout_layer=dict(type='DropPath', drop_prob=0.),
            act_cfg=dict(type='GELU'))

    def _make_channel_att_layer(self, compress_ratio):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.channels * 2, self.channels * 2 // compress_ratio, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels * 2 // compress_ratio, self.channels * 2, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def base_forward1(self, inputs):
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
        out = self.fusion_conv2(torch.cat(outs, dim=1))
        return out

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def forward(self, x1, x2):

        x_sar = self.encoder_sar(x2)
        x_opt = self.encoder_opt(x1)

        out1 = self.base_forward1(x_opt[1:])
        out2 = self.base_forward2(x_sar[1:])

        out_ori = torch.cat([out1, out2], dim=1)
        out_ori = self.discriminator(out_ori)

        out = self.cls_seg(out_ori)

        return out


if __name__ == '__main__':
    model = DualEUNet(3, 2)
    x1 = torch.randn(1, 3, 256, 256)
    x2 = torch.randn(1, 3, 256, 256)
    y = model(x1, x2)
    print(y[0].shape)
