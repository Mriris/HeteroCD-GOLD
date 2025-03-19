import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler


###############################################################################
# 辅助函数
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """返回一个标准化层

    参数:
        norm_type (str) -- 标准化层的名称: batch | instance | none

    对于BatchNorm，我们使用可学习的仿射参数并跟踪运行统计数据（均值/标准差）。
    对于InstanceNorm，我们不使用可学习的仿射参数。我们不跟踪运行统计数据。
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('标准化层 [%s] 未找到' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """返回学习率调度器

    参数:
        optimizer          -- 网络的优化器
        opt (option类) -- 存储所有实验标志；需要是BaseOptions的子类。
                          opt.lr_policy是学习率策略的名称: linear | step | plateau | cosine

    对于'linear'，我们在前<opt.n_epochs>个epoch保持相同的学习率，
    并在接下来的<opt.n_epochs_decay>个epoch线性衰减到零。
    对于其他调度器（step、plateau和cosine），我们使用PyTorch默认的调度器。
    有关更多详细信息，请参见https://pytorch.org/docs/stable/optim.html
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch) / float(opt.n_epochs)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('学习率策略 [%s] 未实现', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """初始化网络权重。

    参数:
        net (network)   -- 要初始化的网络
        init_type (str) -- 初始化方法的名称: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- normal、xavier和orthogonal的缩放因子。

    我们在原始的pix2pix和CycleGAN论文中使用了'normal'。但xavier和kaiming可能对某些应用程序更有效。
    您可以自行尝试。
    """

    def init_func(m):  # 定义初始化函数
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('初始化方法 [%s] 未实现' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm层的权重不是矩阵；只应用正态分布。
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('使用 %s 方法初始化网络' % init_type)
    net.apply(init_func)  # 应用初始化函数<init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """初始化网络：1. 注册CPU/GPU设备（使用GPU时使用多GPU）；2. 初始化网络权重
    参数:
        net (network)      -- 要初始化的网络
        init_type (str)    -- 初始化方法的名称: normal | xavier | kaiming | orthogonal
        gain (float)       -- normal、xavier和orthogonal的缩放因子
        gpu_ids (int list) -- 要使用的GPU id，例如0,1,2

    返回初始化的网络。
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # 多GPU
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02,
             gpu_ids=[]):
    """创建一个生成器

    参数:
        input_nc (int) -- 输入图像的通道数
        output_nc (int) -- 输出图像的通道数
        ngf (int) -- 最后一个卷积层中的过滤器数量
        netG (str) -- 生成器的架构: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- 要使用的标准化层的类型: batch | instance | none
        use_dropout (bool) -- 是否在生成器中使用dropout
        init_type (str)    -- 初始化方法的名称: normal | xavier | kaiming | orthogonal
        init_gain (float)  -- normal、xavier和orthogonal的缩放因子
        gpu_ids (int list) -- 要使用的GPU id，例如0,1,2

    返回一个生成器

    我们的当前实现提供了两种类型的生成器：
        U-Net: [unet_128] (用于128x128输入图像) 和 [unet_256] (用于256x256输入图像)
        生成器由多个下采样/上采样层（Unet）和几个残差块组成（残差net）。

    用于批量标准化和其他标准化类型的选项（例如，instance规范化）。
        - 对于数据较少的数据集（例如，少于500张训练图像）使用--batch_size=1会有更好的表现 [参见论文中的讨论]
        - 如果使用--batch_size=1，建议使用'instance'标准化。
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('生成器模型名称 [%s] 未实现' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """创建一个判别器

    参数:
        input_nc (int)     -- 输入图像的通道数
        ndf (int)          -- 第一个卷积层中的过滤器数量
        netD (str)         -- 判别器的架构: basic | n_layers | pixel
        n_layers_D (int)   -- 判别器中要使用的卷积层数；仅当netD=='n_layers'时有效
        norm (str)         -- 要使用的标准化层的类型: batch | instance | none
        init_type (str)    -- 初始化方法的名称: normal | xavier | kaiming | orthogonal
        init_gain (float)  -- normal、xavier和orthogonal的缩放因子。
        gpu_ids (int list) -- 要使用的GPU id，例如0,1,2

    返回一个判别器

    当前版本仅实现了三种类型的判别器：
        [basic]: 'PatchGAN'判别器--可用于大多数不同类型的计算机视觉任务
        [n_layers]: 具有n个卷积层的更多PatchGAN判别器
        [pixel]: 1x1 PixelGAN判别器--这样一个简单的判别器每次只查看一个像素

    基本判别器结构最初在论文[Pix2Pix]的9x9 PatchGAN中描述：
    Phillip Isola等人，"使用条件对抗网络的图像到图像转换"。CVPR 2017
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # 默认PatchGAN分类器
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # 可以指定更多层的PatchGAN
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':  # 分类单个像素的判别器
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('判别器模型名称 [%s] 未实现' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# 类
##############################################################################
class GANLoss(nn.Module):
    """定义不同GAN目标的GAN损失。

    目标可以是'vanilla'或'lsgan'。
    vanilla GAN损失在[GAN]论文中提出:
    https://arxiv.org/abs/1406.2661
    LSGAN损失在[LSGAN]论文中提出:
    https://arxiv.org/abs/1611.04076

    的GANs的目标是将mse调整到真实标签值（即1）。
    vanilla GANs的目标是获得真实标签值的对数概率（即0）。
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ 初始化GANLoss类。

        参数:
            gan_mode (str) - - 要使用的gan目标；可以是vanilla、lsgan或wgangp
            target_real_label (bool) - - 真实图像/视觉的标签值
            target_fake_label (bool) - - 假图像/视觉的标签值

        注意：不要使用sigmoid作为GAN的最后一层，因为通过BCEWithLogitsLoss计算损失。
        使用BCELoss时需要sigmoid。
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan模式 %s 未实现' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """使用与预测相同类型和形状的张量创建标签张量。"""
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """计算损失。

        参数:
            prediction (tensor) - - 通常是判别器模型的输出
            target_is_real (bool) - - 真实标签为真，假标签为假。

        返回:
            基于预测和目标的损失。
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """计算WGAN GP项中的梯度惩罚。"""
    if lambda_gp > 0.0:
        if type == 'real':  # 真实图像上的梯度惩罚
            interpolatesv = real_data
        elif type == 'fake':  # 假图像上的梯度惩罚
            interpolatesv = fake_data
        elif type == 'mixed':  # 真实和假图像的随机插值
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(
                *real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} 不是一个实现的梯度惩罚类型'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # 平坦化梯度
        gradient_penalty = (((gradients + 1e-16).norm(2,
                                                      dim=1) - constant) ** 2).mean() * lambda_gp  # 添加一个小的常数以防止网络传播的数值不稳定性
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """使用多个残差块的Resnet生成器。

    我们构造一个Resnet生成器，由几个下采样/上采样操作和9个残差块组成。
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        """构建ResNet生成器

        参数:
            input_nc (int)      -- 输入图像的通道数
            output_nc (int)     -- 输出图像的通道数
            ngf (int)           -- 最后一个卷积层中的过滤器数量
            norm_layer          -- 标准化层
            use_dropout (bool)  -- 如果使用dropout层
            n_blocks (int)      -- 生成器中resnet块的数量
            padding_type (str)  -- 卷积层中使用的padding类型: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # 添加下采样层
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # 添加ResNet块

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):  # 添加上采样层
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """标准前向"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """定义Resnet的一个块"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """初始化ResNet块

        一个resnet块是残差网络，包含两个带有标准化层的3x3卷积层。

        我们构造一个con块，由一个conv层、一个规范化层和一个dropout层组成（如果适用），
        然后是第二个conv层和一个规范化层。

        我们实现了skip连接的残差块。

        残差块原始论文: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """构造一个卷积块。

        参数:
            dim (int)           -- 卷积层中过滤器的数量
            padding_type (str)  -- 卷积层中使用的padding类型: reflect | replicate | zero
            norm_layer          -- 标准化层
            use_dropout (bool)  -- 是否使用dropout层
            use_bias (bool)     -- 卷积层是否使用偏置项

        返回一个卷积块（由卷积层、标准化层和激活层组成）。
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] 未实现' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] 未实现' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """前向函数（带跳过连接）"""
        out = x + self.conv_block(x)  # 添加跳过连接
        return out


class UnetGenerator(nn.Module):
    """使用跳过连接的U-Net生成器

    来自论文："使用条件对抗网络的图像到图像转换"
    其中原始U-Net来自论文："用于生物医学图像分割的U-Net"
    """

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """构造U-Net生成器
        参数:
            input_nc (int)  -- 输入图像的通道数
            output_nc (int) -- 输出图像的通道数
            num_downs (int) -- UNet中的下采样层数。例如，如果|num_downs| == 7,
                                图像的大小为 128x128 -> 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4 -> 2x2 -> 1x1。
            ngf (int)       -- 最后一个卷积层中的过滤器数量
            norm_layer      -- 标准化层

        我们构造U-Net从最外层到最内层，
        再从最内层到最外层。
        这是实现跳过连接的一种优雅方式
        """
        super(UnetGenerator, self).__init__()
        # 构造unet结构
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)  # 添加最里面的层
        # 添加中间层，逐渐减少通道数
        for i in range(num_downs - 5):  # 添加中间层
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        # 逐渐恢复分辨率
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)  # 添加最外层

    def forward(self, input):
        """标准前向"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """用于UnetGenerator的Unet跳过连接"""

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """构造Unet跳过连接

        参数:
            outer_nc (int) -- 输出层中的过滤器数量
            inner_nc (int) -- 内层中的过滤器数量
            input_nc (int) -- 输入层中的过滤器数量
            submodule (UnetSkipConnectionBlock) -- 之前定义的子模块
            outermost (bool)    -- 如果此模块是最外层
            innermost (bool)    -- 如果此模块是最内层
            norm_layer          -- 标准化层
            use_dropout (bool)  -- 是否使用dropout层
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # 添加跳过连接
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """PatchGAN判别器 - 如Pix2Pix论文中所述"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """构造PatchGAN判别器

        参数:
            input_nc (int)  -- 输入图像的通道数
            ndf (int)       -- 第一个卷积层中的过滤器数量
            n_layers (int)  -- 判别器中卷积层的数量
            norm_layer      -- 标准化层
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # 没有偏置项的batch-norm
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # 逐渐增加通道数量
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # 输出一个通道
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """标准前向。"""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """定义一个1x1 PatchGAN判别器(pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """构造一个1x1 PatchGAN判别器

        参数:
            input_nc (int)  -- 输入图像的通道数
            ndf (int)       -- 最后一个卷积层中的过滤器数量
            norm_layer      -- 标准化层
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # 没有偏置项的batch-norm
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """标准前向传播"""
        return self.net(input)
