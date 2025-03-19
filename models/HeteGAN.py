import torch
from .base_model import BaseModel
from . import networks
from .DualEUNet import DualEUNet
from .loss import *


class Pix2PixModel(BaseModel):
    """ 该类实现了pix2pix模型，用于学习从输入图像到输出图像的映射，基于成对数据。

    模型训练需要'--dataset_mode aligned'数据集。
    默认情况下，它使用'--netG unet256' U-Net生成器，
    '--netD basic'判别器(PatchGAN)，
    以及'--gan_mode' vanilla GAN损失(原始GAN论文中使用的交叉熵目标)。

    pix2pix论文: https://arxiv.org/pdf/1611.07004.pdf
    """

    def __init__(self, opt, is_train=True):
        """初始化pix2pix类。

        参数:
            opt (Option类) -- 存储所有实验标志的类；需要是BaseOptions的子类
        """
        BaseModel.__init__(self, opt, is_train=True)
        # 指定要打印的训练损失。训练/测试脚本将调用<BaseModel.get_current_losses>
        self.loss_names = ['CD']
        # 指定要保存/显示的图像。训练/测试脚本将调用<BaseModel.get_current_visuals>
        self.change_pred = None
        self.isTrain = is_train
        # 指定要保存到磁盘的模型。训练/测试脚本将调用<BaseModel.save_networks>和<BaseModel.load_networks>
        self.model_names = ['CD']
        # 定义网络(包括生成器和判别器)
        self.netCD = DualEUNet(3, 2)
        self.netCD.to(opt.gpu_ids[0])
        self.is_train = is_train
        if is_train:
            self.netCD = torch.nn.DataParallel(self.netCD, opt.gpu_ids)  # 多GPU支持

        if self.isTrain:
            self.optimizer_G = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.netCD.parameters()), lr=opt.lr,
                                                 betas=(0.9, 0.999), weight_decay=0.01)
            self.optimizers.append(self.optimizer_G)

    def set_input(self, A, B, label, name, device):
        """从数据加载器解包输入数据并执行必要的预处理步骤。

        参数:
            input (dict): 包含数据本身及其元数据信息。

        选项'direction'可用于交换域A和域B中的图像。
        """

        self.opt_img = A.to(device)
        self.sar_img = B.to(device)
        self.label = label.to(device)
        self.name = name

    def load_weights(self, checkpoint_path):
        """加载模型权重

        参数:
            checkpoint_path (str): 权重文件的路径
        """
        checkpoint = torch.load(checkpoint_path)
        for key in list(checkpoint.keys()):
            if key.startswith('module.'):
                checkpoint[key[7:]] = checkpoint[key]
                del checkpoint[key]
        self.netCD.load_state_dict(checkpoint)
        if not self.isTrain:
            self.netCD.eval()
            print("已加载模型权重，并设置为评估模式")

    def forward(self):
        """运行前向传播；由<optimize_parameters>和<test>函数调用。"""
        [self.fake_B, self.fake_BB] = self.netCD(self.real_A, self.real_B)  # G(A)

    def get_val_pred(self):
        """获取验证集的预测结果

        返回:
            tuple: 包含预测结果和相应的损失
        """
        self.netCD.eval()
        self.is_train = False
        with torch.no_grad():
            self.forward_CD()
            cls_weights = torch.tensor([0.2, 0.8]).cuda()
            loss_bn = CE_Loss(self.change_pred, self.label, cls_weights)
        self.is_train = True
        return self.change_pred, loss_bn

    def forward_CD(self):
        """执行变化检测的前向传播"""
        self.change_pred = self.netCD(self.opt_img, self.sar_img)  # G(A)

    def backward_G(self):
        """计算生成器的损失并进行反向传播"""
        self.change_pred = F.interpolate(self.change_pred, size=(self.opt_img.size(2), self.opt_img.size(3)),
                                         mode='bilinear', align_corners=True)
        cls_weights = torch.tensor([0.2, 0.8]).cuda()
        self.label = self.label.long()
        self.loss_CD = CE_Loss(self.change_pred, self.label, cls_weights=cls_weights) * 100 + Dice_loss(
            self.change_pred, self.label) * 100

        # 组合损失并计算梯度
        self.loss_G = self.loss_CD
        self.loss_G.backward()

    def optimize_parameters(self, epoch):
        """优化模型参数

        参数:
            epoch (int): 当前训练轮次

        返回:
            tensor: 变化检测的预测结果
        """
        self.netCD.train()
        self.forward_CD()
        self.optimizer_G.zero_grad()  # 将G的梯度设为零
        self.backward_G()  # 计算G的梯度
        self.optimizer_G.step()
        return self.change_pred
