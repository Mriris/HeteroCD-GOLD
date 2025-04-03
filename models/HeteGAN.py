import torch
from .base_model import BaseModel
from . import networks
from .DualEUNet import DualEUNet
from .loss import *
import os
from collections import OrderedDict


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

    def save_networks(self, epoch, save_best=False):
        """将所有网络保存到磁盘，覆盖基类方法以支持保存最佳模型

        参数:
            epoch (int/str) -- 当前epoch或'best'等标识
            save_best (bool) -- 是否将模型保存为最佳模型
        """
        for name in self.model_names:
            if isinstance(name, str):
                if save_best:
                    save_filename = 'best_net_%s.pth' % (name)
                else:
                    save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)
                    
                print(f'模型已保存: {save_path}')
                
    def load_networks(self, epoch):
        """从磁盘加载所有网络，覆盖基类方法以支持加载特定epoch或最佳模型

        参数:
            epoch (int/str) -- 当前epoch或'best'/'latest'等标识
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                
                # 检查文件是否存在，如果不存在且是尝试加载best模型，尝试找到最新的模型
                if not os.path.exists(load_path) and epoch == 'best':
                    print(f"未找到最佳模型 {load_path}，尝试加载最新模型...")
                    # 查找最新的模型
                    model_files = [f for f in os.listdir(self.save_dir) if f.endswith(f'_net_{name}.pth') and not f.startswith('best')]
                    if model_files:
                        # 按文件名排序，通常文件名格式为: <epoch>_net_<name>.pth
                        model_files.sort(key=lambda x: int(x.split('_')[0]) if x.split('_')[0].isdigit() else -1, reverse=True)
                        load_filename = model_files[0]
                        load_path = os.path.join(self.save_dir, load_filename)
                        print(f"将加载最新模型: {load_path}")
                    else:
                        print(f"未找到任何可用模型")
                        continue
                
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('从%s加载网络' % load_path)
                
                try:
                    state_dict = torch.load(load_path, map_location=str(self.device))
                    
                    # 处理metadata
                    if hasattr(state_dict, '_metadata'):
                        del state_dict._metadata

                    # 尝试移除"module."前缀
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        if k.startswith('module.'):
                            name = k[7:]  # 移除 'module.' 前缀
                        else:
                            name = k
                        new_state_dict[name] = v
                    
                    # 加载状态字典
                    net.load_state_dict(new_state_dict)
                    print(f"模型 {name} 加载成功")
                except Exception as e:
                    print(f"在加载 {load_path} 时出错: {e}")
                    # 尝试使用更健壮的加载方法
                    try:
                        # 更安全的加载方法：只加载模型中存在的参数
                        model_dict = net.state_dict()
                        # 过滤state_dict，只保留存在于模型中的项
                        state_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
                        model_dict.update(state_dict)
                        net.load_state_dict(model_dict)
                        print(f"使用部分加载方式成功加载模型 {name}")
                    except Exception as e2:
                        print(f"备用加载方法也失败: {e2}")
