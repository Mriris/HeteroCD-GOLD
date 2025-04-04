import torch
from .base_model import BaseModel
from . import networks
from .DualEUNet import DualEUNet, TripleEUNet
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
        self.loss_names = ['CD', 'Distill']
        # 指定要保存/显示的图像。训练/测试脚本将调用<BaseModel.get_current_visuals>
        self.change_pred = None
        self.teacher_pred = None
        self.isTrain = is_train
        # 指定要保存到磁盘的模型。训练/测试脚本将调用<BaseModel.save_networks>和<BaseModel.load_networks>
        self.model_names = ['CD']
        # 是否使用三分支网络和蒸馏学习
        self.use_distill = getattr(opt, 'use_distill', True)
        
        # 定义网络
        if self.use_distill:
            self.netCD = TripleEUNet(3, 2)
            # 初始化蒸馏损失
            self.distill_loss = MultiLevelDistillationLoss(
                feature_weight=0.3, 
                output_weight=0.7, 
                temperature=2.5
            )
        else:
            self.netCD = DualEUNet(3, 2)
            
        self.netCD.to(opt.gpu_ids[0])
        self.is_train = is_train
        
        if is_train:
            self.netCD = torch.nn.DataParallel(self.netCD, opt.gpu_ids)  # 多GPU支持

        if self.isTrain:
            self.optimizer_G = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.netCD.parameters()), lr=opt.lr,
                                                 betas=(0.9, 0.999), weight_decay=0.01)
            self.optimizers.append(self.optimizer_G)

    def set_input(self, A, B, label, name, device, C=None):
        """从数据加载器解包输入数据并执行必要的预处理步骤。

        参数:
            A (tensor): 时间点1的光学图像
            B (tensor): 时间点2的SAR图像
            label (tensor): 变化检测标签
            name (str): 图像名称
            device (torch.device): 设备
            C (tensor, optional): 时间点2的光学图像，用于教师网络
        """
        self.opt_img = A.to(device)
        self.sar_img = B.to(device)
        self.label = label.to(device)
        self.name = name
        
        # 如果提供了时间点2的光学图像且使用蒸馏学习，则存储它
        if C is not None and self.use_distill:
            self.opt_img2 = C.to(device)
        else:
            self.opt_img2 = None

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
            if self.use_distill and self.opt_img2 is not None:
                # 对于TripleEUNet，只获取学生网络的输出
                self.change_pred = self.netCD(self.opt_img, self.sar_img, self.opt_img2, is_training=False)
            else:
                self.forward_CD()
                
            # 使用与训练一致的类权重
            cls_weights = torch.tensor([0.1, 0.9]).cuda()
            loss_bn = CE_Loss(self.change_pred, self.label, cls_weights)
            
        self.is_train = True
        return self.change_pred, loss_bn
        
    def get_teacher_pred(self):
        """获取教师网络的预测结果，仅用于验证

        返回:
            tuple: 包含教师网络预测结果和相应的损失
        """
        if not self.use_distill or self.opt_img2 is None:
            return None, 0.0
            
        self.netCD.eval()
        with torch.no_grad():
            # 使用第一个光学图像和第二个光学图像进行预测
            student_out, teacher_out, _, _, _, _ = self.netCD(
                self.opt_img, self.opt_img2, self.opt_img2, is_training=True
            )
            self.teacher_pred = teacher_out
            
            # 使用与训练一致的类权重
            cls_weights = torch.tensor([0.1, 0.9]).cuda()
            loss_bn = CE_Loss(self.teacher_pred, self.label, cls_weights)
            
        return self.teacher_pred, loss_bn

    def forward_CD(self):
        """执行变化检测的前向传播"""
        if self.use_distill and self.opt_img2 is not None and self.is_train:
            # 使用三分支网络进行训练
            self.student_out, self.teacher_out, self.student_feat, self.teacher_feat, self.student_mid_feat, self.teacher_mid_feat = self.netCD(
                self.opt_img, self.sar_img, self.opt_img2, is_training=True
            )
            self.change_pred = self.student_out
        else:
            # 使用双分支网络或者三分支网络的测试模式
            self.change_pred = self.netCD(self.opt_img, self.sar_img)

    def backward_G(self):
        """计算生成器的损失并进行反向传播"""
        self.change_pred = F.interpolate(self.change_pred, size=(self.opt_img.size(2), self.opt_img.size(3)),
                                         mode='bilinear', align_corners=True)
        # 调整类权重，大幅提高类别1(变化区域)的权重
        cls_weights = torch.tensor([0.1, 0.9]).cuda()
        self.label = self.label.long()
        
        # 主要变化检测损失
        self.loss_CD = CE_Loss(self.change_pred, self.label, cls_weights=cls_weights) * 100 + Dice_loss(
            self.change_pred, self.label) * 100
            
        # 初始化蒸馏损失为0
        self.loss_Distill = torch.tensor(0.0).cuda()
        
        # 如果使用蒸馏学习且有教师网络输出
        if self.use_distill and hasattr(self, 'teacher_out') and self.teacher_out is not None:
            # 调整教师网络输出尺寸以匹配学生网络
            teacher_out_resized = F.interpolate(
                self.teacher_out, 
                size=(self.change_pred.size(2), self.change_pred.size(3)),
                mode='bilinear', 
                align_corners=True
            )
            
            # 创建特征掩码 - 重点关注类别1的区域
            # 确保维度匹配：self.label形状为[B, 1, H, W]或[B, H, W]
            if len(self.label.shape) == 3:  # 形状为[B, H, W]
                label_mask = self.label.unsqueeze(1)  # 变为[B, 1, H, W]
            else:
                label_mask = self.label  # 已经是[B, 1, H, W]
                
            feature_mask = torch.zeros_like(label_mask, dtype=torch.float)
            # 将类别1区域的权重设置得更高，以增强对变化区域的学习
            feature_mask[label_mask == 1] = 3.0
            # 将类别0区域设置较低权重但不为0
            feature_mask[label_mask == 0] = 0.5
            
            feature_mask = F.interpolate(
                feature_mask, 
                size=self.student_feat.size()[2:],
                mode='nearest'
            )
            
            # 计算蒸馏损失，但降低权重
            self.loss_Distill, feat_loss, out_loss = self.distill_loss(
                self.student_feat,
                self.teacher_feat,
                self.change_pred,
                teacher_out_resized,
                feature_mask
            )
            # 将蒸馏损失权重从10降低到2
            self.loss_Distill = self.loss_Distill * 2.0

        # 组合损失并计算梯度
        self.loss_G = self.loss_CD + self.loss_Distill
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
                # 如果epoch是完整的文件名（例如来自training_info.txt）
                if isinstance(epoch, str) and epoch.endswith('_net_CD.pth'):
                    load_filename = epoch
                else:
                    load_filename = '%s_net_%s.pth' % (epoch, name)
                    
                load_path = os.path.join(self.save_dir, load_filename)
                
                # 检查文件是否存在
                if not os.path.exists(load_path):
                    print(f"警告: 模型文件 {load_path} 不存在！")
                    
                    # 尝试不同的情况
                    if epoch == 'latest' or (isinstance(epoch, str) and epoch.endswith('_net_CD.pth')):
                        print(f"尝试查找最新的模型文件...")
                        # 收集所有相关的模型文件
                        model_files = [f for f in os.listdir(self.save_dir) 
                                       if f.endswith(f'_net_{name}.pth') and not f.startswith('best')]
                        
                        if model_files:
                            # 根据文件名中的数字（通常是epoch）排序
                            model_files.sort(key=lambda x: int(x.split('_')[0]) 
                                            if x.split('_')[0].isdigit() else -1, 
                                            reverse=True)
                            
                            load_filename = model_files[0]
                            load_path = os.path.join(self.save_dir, load_filename)
                            print(f"将加载最新模型: {load_path}")
                        else:
                            print(f"未找到任何普通模型文件，尝试查找最佳模型...")
                            # 寻找最佳模型
                            best_model = [f for f in os.listdir(self.save_dir) 
                                         if f.startswith('best_net_') and f.endswith('.pth')]
                            if best_model:
                                load_filename = best_model[0]
                                load_path = os.path.join(self.save_dir, load_filename)
                                print(f"将加载最佳模型: {load_path}")
                            else:
                                print(f"警告：未找到任何可用模型文件！将使用初始化的模型。")
                                continue
                    elif epoch == 'best':
                        # 尝试加载best模型
                        print(f"尝试查找最佳模型文件...")
                        best_model = [f for f in os.listdir(self.save_dir) 
                                     if f.startswith('best_net_') and f.endswith('.pth')]
                        if best_model:
                            load_filename = best_model[0]
                            load_path = os.path.join(self.save_dir, load_filename)
                            print(f"将加载最佳模型: {load_path}")
                        else:
                            # 如果没有最佳模型，尝试加载最新模型
                            print(f"未找到最佳模型，尝试查找最新模型...")
                            model_files = [f for f in os.listdir(self.save_dir) 
                                          if f.endswith(f'_net_{name}.pth')]
                            if model_files:
                                model_files.sort(key=lambda x: int(x.split('_')[0]) 
                                                if x.split('_')[0].isdigit() else -1, 
                                                reverse=True)
                                load_filename = model_files[0]
                                load_path = os.path.join(self.save_dir, load_filename)
                                print(f"将加载最新模型: {load_path}")
                            else:
                                print(f"警告：未找到任何可用模型文件！将使用初始化的模型。")
                                continue
                    else:
                        # 如果是特定epoch但找不到，尝试最佳模型或最新模型
                        print(f"无法找到epoch {epoch}的模型文件，尝试查找其他可用模型...")
                        
                        # 先尝试加载最佳模型
                        best_model = [f for f in os.listdir(self.save_dir) 
                                     if f.startswith('best_net_') and f.endswith('.pth')]
                        if best_model:
                            load_filename = best_model[0]
                            load_path = os.path.join(self.save_dir, load_filename)
                            print(f"将加载最佳模型: {load_path}")
                        else:
                            # 再尝试加载最新的普通模型
                            model_files = [f for f in os.listdir(self.save_dir) 
                                          if f.endswith(f'_net_{name}.pth')]
                            if model_files:
                                model_files.sort(key=lambda x: int(x.split('_')[0]) 
                                                if x.split('_')[0].isdigit() else -1, 
                                                reverse=True)
                                load_filename = model_files[0]
                                load_path = os.path.join(self.save_dir, load_filename)
                                print(f"将加载最新模型: {load_path}")
                            else:
                                print(f"警告：未找到任何可用模型文件！将使用初始化的模型。")
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
                    print(f"成功加载模型！")
                    
                    # 检查加载的参数是否有效
                    param_sum = sum(p.sum().item() for p in net.parameters() if p.requires_grad)
                    print(f"模型参数总和: {param_sum:.4f}")
                    if abs(param_sum) <= 0.1:
                        print("警告：模型参数总和接近零，可能未正确加载！")
                except Exception as e:
                    print(f"加载模型时出错: {e}")
                    print(f"无法加载模型，将使用初始化的权重！")
