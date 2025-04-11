from .base_model import BaseModel
import math
import os
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F

from .DualEUNet import DualEUNet, TripleEUNet
from .base_model import BaseModel
from .loss import *
from .loss import HeterogeneousAttentionDistillationLoss, DifferenceAttentionLoss, \
    DynamicHeterogeneousWeightTransferLoss


class TripleHeteCD(BaseModel):
    def __init__(self, opt, is_train=True):
        """初始化

        参数:
            opt (Option类) -- 存储所有实验标志的类；需要是BaseOptions的子类
        """
        BaseModel.__init__(self, opt, is_train=True)

        # 是否使用三分支网络和蒸馏学习
        self.use_distill = opt.use_distill

        # 添加动态权重分配相关参数
        self.use_dynamic_weights = opt.use_dynamic_weights  # 直接从opt中获取参数值
        self.weight_warmup_epochs = opt.weight_warmup_epochs  # 权重热身阶段的轮次数
        self.current_epoch = 0  # 当前训练轮次

        # 初始权重设置
        self.init_cd_weight = opt.init_cd_weight
        self.init_distill_weight = opt.init_distill_weight
        self.init_diff_att_weight = opt.init_diff_att_weight

        # 指定要打印的训练损失。训练/测试脚本将调用<BaseModel.get_current_losses>
        self.loss_names = ['CD']  # 基础损失名称
        if self.use_distill:
            self.loss_names.extend(['Distill', 'Diff_Att'])
            if self.use_dynamic_weights:
                self.loss_names.append('Dynamic_Weight')
                # 初始化权重不确定性参数 (可学习参数)
                self.log_vars = nn.Parameter(torch.zeros(3))  # 为CD、Distill和Diff_Att损失创建可学习的权重参数
                # 只在启用动态权重时添加动态异源权重迁移损失
                self.dynamic_weight_loss = DynamicHeterogeneousWeightTransferLoss(
                    reduction='mean'
                )
            else:
                self.log_vars = None
                self.dynamic_weight_loss = None
        else:
            self.log_vars = None
            self.dynamic_weight_loss = None

        # 指定要保存/显示的图像。训练/测试脚本将调用<BaseModel.get_current_visuals>
        self.change_pred = None
        self.teacher_pred = None
        self.isTrain = is_train
        # 指定要保存到磁盘的模型。训练/测试脚本将调用<BaseModel.save_networks>和<BaseModel.load_networks>
        self.model_names = ['CD']

        # 定义网络
        if self.use_distill:
            self.netCD = TripleEUNet(3, 2)
            # 使用新的异源注意力蒸馏损失
            self.distill_loss = HeterogeneousAttentionDistillationLoss(
                feature_weight=getattr(opt, 'distill_alpha', 0.3),
                output_weight=getattr(opt, 'distill_beta', 0.5),
                diff_att_weight=getattr(opt, 'distill_gamma', 0.2),
                temperature=getattr(opt, 'distill_temp', 2.5),
                reduction=getattr(opt, 'kl_div_reduction', 'batchmean')
            )
            # 额外添加差异图注意力迁移损失
            self.diff_att_loss = DifferenceAttentionLoss(
                reduction='mean',
                alpha=1.0,
                beta=0.5,
                gamma=0.5
            )
        else:
            self.netCD = DualEUNet(3, 2)

        self.netCD.to(opt.gpu_ids[0])
        self.is_train = is_train

        if is_train:
            self.netCD = torch.nn.DataParallel(self.netCD, opt.gpu_ids)  # 多GPU支持

        if self.isTrain:
            # 将log_vars添加到优化器中
            params = [
                {'params': filter(lambda p: p.requires_grad, self.netCD.parameters())},
            ]
            if self.use_dynamic_weights and self.log_vars is not None:
                params.append({'params': self.log_vars, 'lr': opt.lr * 0.1})  # 使用较小的学习率优化权重参数
            self.optimizer_G = torch.optim.AdamW(params, lr=opt.lr,
                                                 betas=(0.9, 0.999), weight_decay=0.01)
            self.optimizers.append(self.optimizer_G)

    def set_epoch(self, epoch):
        """设置当前训练轮次，用于动态权重计算

        参数:
            epoch (int): 当前训练轮次
        """
        self.current_epoch = epoch

    def get_dynamic_weights(self):
        """计算基于不确定性的动态权重

        返回:
            tuple: (CD损失权重, 蒸馏损失权重, 差异图注意力损失权重)
        """
        # 基于不确定性的权重计算
        # 参考: "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
        if self.use_dynamic_weights:
            # 使用softplus确保数值稳定性，替代直接使用指数
            precision = torch.nn.functional.softplus(-self.log_vars) + 1e-8

            # 引入变化区域关注因子 - 提高对CD损失的注重度
            # 变化区域通常是少数类，需要更高的权重来平衡
            cd_focus_factor = 1.2
            precision[0] = precision[0] * cd_focus_factor  # 增加CD损失的权重

            # 应用热身和训练进度调整
            if self.current_epoch < self.weight_warmup_epochs:
                # 热身阶段: 从固定权重逐渐过渡到动态权重
                progress = self.current_epoch / self.weight_warmup_epochs
                # 固定的初始权重
                fixed_weights = torch.tensor([self.init_cd_weight, self.init_distill_weight, self.init_diff_att_weight],
                                             device=self.log_vars.device)

                # 平滑过渡：使用更平滑的函数 - cos函数而不是线性
                alpha = 0.5 * (1 - math.cos(progress * math.pi))

                # 归一化固定权重
                fixed_weights = fixed_weights / fixed_weights.sum()

                # 混合固定权重和动态权重
                weights = (1 - alpha) * fixed_weights + alpha * (precision / precision.sum())
            else:
                # 训练后期：添加自适应衰减机制，逐渐增加CD损失权重
                late_stage_factor = min(1.0 + (self.current_epoch - self.weight_warmup_epochs) / 100.0, 1.5)
                precision[0] = precision[0] * late_stage_factor
                weights = precision / precision.sum()

            # 重新缩放权重到原始量级
            scale_factors = torch.tensor([self.init_cd_weight, self.init_distill_weight, self.init_diff_att_weight],
                                         device=self.log_vars.device)
            weights = weights * scale_factors.sum()
            scaled_weights = weights * scale_factors

            return scaled_weights[0].item(), scaled_weights[1].item(), scaled_weights[2].item()
        else:
            # 使用固定权重
            return self.init_cd_weight, self.init_distill_weight, self.init_diff_att_weight

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
            student_out, teacher_out, _, _, _, _, _, _, _ = self.netCD(
                self.opt_img, self.sar_img, self.opt_img2, is_training=True
            )
            self.teacher_pred = teacher_out

            # 使用与训练一致的类权重
            cls_weights = torch.tensor([0.1, 0.9]).cuda()
            loss_bn = CE_Loss(self.teacher_pred, self.label, cls_weights)

        return self.teacher_pred, loss_bn

    def forward_CD(self):
        """执行变化检测的前向传播"""
        if self.use_distill and self.opt_img2 is not None and self.is_train:
            # 使用增强的三分支网络进行训练，返回值包括原始特征
            # (学生输出, 教师输出, 学生增强特征, 教师增强特征, 学生中间特征, 教师中间特征, 光学t1特征, 光学t2特征, SAR t2特征)
            self.student_out, self.teacher_out, self.student_feat, self.teacher_feat, \
                self.student_mid_feat, self.teacher_mid_feat, self.opt_t1_feat, \
                self.opt_t2_feat, self.sar_t2_feat = self.netCD(
                self.opt_img, self.sar_img, self.opt_img2, is_training=True
            )
            self.change_pred = self.student_out
        else:
            # 使用双分支网络或者三分支网络的测试模式
            self.change_pred = self.netCD(self.opt_img, self.sar_img)
        
        return self.change_pred

    def compute_losses(self):
        """计算损失但不执行反向传播，用于与混合精度训练配合使用"""
        self.change_pred = F.interpolate(self.change_pred, size=(self.opt_img.size(2), self.opt_img.size(3)),
                                         mode='bilinear', align_corners=True)
        # 调整类权重，进一步提高类别1(变化区域)的权重
        cls_weights = torch.tensor([0.1, 0.9]).cuda()
        self.label = self.label.long()

        # 主要变化检测损失 - 增加类别平衡焦点损失成分，提高对变化区域的关注
        ce_loss = CE_Loss(self.change_pred, self.label, cls_weights=cls_weights)
        dice_loss = Dice_loss(self.change_pred, self.label)

        # 添加手动类别重要性调整，进一步关注变化区域
        self.loss_CD = ce_loss * 100 + dice_loss * 150  # 增加Dice损失权重，因为它对小区域更敏感

        # 初始化蒸馏损失和差异图注意力损失为0
        self.loss_Distill = torch.tensor(0.0).cuda()
        self.loss_Diff_Att = torch.tensor(0.0).cuda()

        # 只在启用动态权重时初始化Dynamic_Weight损失
        if self.use_dynamic_weights and self.dynamic_weight_loss is not None:
            self.loss_Dynamic_Weight = torch.tensor(0.0).cuda()

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
            feature_mask[label_mask == 1] = 5.0  # 增加变化区域权重
            # 将类别0区域设置较低权重但不为0
            feature_mask[label_mask == 0] = 0.3  # 降低非变化区域权重

            feature_mask = F.interpolate(
                feature_mask,
                size=self.student_feat.size()[2:],
                mode='nearest'
            )

            # 计算差异图注意力损失 - 使用原始特征而不是增强后的特征
            diff_att_total, diff_att_loss, channel_att_loss, spatial_att_loss = self.diff_att_loss(
                self.student_feat, self.teacher_feat,
                self.opt_t1_feat, self.opt_t2_feat, self.sar_t2_feat
            )

            # 只在启用动态权重时计算动态异源权重迁移损失
            if self.use_dynamic_weights and self.dynamic_weight_loss is not None:
                dynamic_weight_total, dynamic_feat_loss, dynamic_att_loss, dynamic_diff_loss = self.dynamic_weight_loss(
                    self.student_feat, self.teacher_feat,
                    self.opt_t1_feat, self.opt_t2_feat, self.sar_t2_feat
                )
                self.loss_Dynamic_Weight = dynamic_weight_total

            # 计算异源注意力蒸馏损失
            distill_total, feat_loss, out_loss, _ = self.distill_loss(
                self.student_feat,
                self.teacher_feat,
                self.change_pred,
                teacher_out_resized,
                self.opt_t1_feat,
                self.opt_t2_feat,
                self.sar_t2_feat,
                feature_mask
            )

            # 记录原始损失值（用于显示）
            self.loss_CD_orig = self.loss_CD.clone().detach()
            self.loss_Distill = distill_total
            self.loss_Diff_Att = diff_att_total

            # 应用权重
            if self.use_dynamic_weights and self.log_vars is not None:
                # 获取动态权重
                cd_weight, distill_weight, diff_att_weight = self.get_dynamic_weights()
                dynamic_weight = diff_att_weight * 0.5  # 设置动态异源权重迁移损失权重为差异图注意力损失权重的一半

                self.loss_CD = self.loss_CD * cd_weight
                self.loss_Distill = self.loss_Distill * distill_weight
                self.loss_Diff_Att = self.loss_Diff_Att * diff_att_weight
                if hasattr(self, 'loss_Dynamic_Weight'):
                    self.loss_Dynamic_Weight = self.loss_Dynamic_Weight * dynamic_weight

                # # 打印当前权重（仅用于调试）
                # if self.current_epoch % 10 == 0 and torch.cuda.current_device() == 0:
                #     print(f"当前动态权重: CD={cd_weight:.2f}, Distill={distill_weight:.2f}, Diff_Att={diff_att_weight:.2f}, Dynamic={dynamic_weight:.2f}")
            else:
                # 使用固定权重
                self.loss_CD = self.loss_CD * self.init_cd_weight
                self.loss_Distill = self.loss_Distill * self.init_distill_weight
                self.loss_Diff_Att = self.loss_Diff_Att * self.init_diff_att_weight
        else:
            # 如果不使用蒸馏学习，则只使用CD损失
            if self.use_dynamic_weights and self.log_vars is not None:
                cd_weight, _, _ = self.get_dynamic_weights()
                self.loss_CD = self.loss_CD * cd_weight
            else:
                self.loss_CD = self.loss_CD * self.init_cd_weight

        # 组合损失
        self.loss_G = self.loss_CD + self.loss_Distill + self.loss_Diff_Att
        if self.use_dynamic_weights and hasattr(self, 'loss_Dynamic_Weight'):
            self.loss_G += self.loss_Dynamic_Weight
            
        return self.loss_G

    def backward_G(self):
        """计算生成器的损失并进行反向传播"""
        self.compute_losses()  # 计算损失
        self.loss_G.backward()  # 反向传播

    def optimize_parameters(self, epoch):
        """优化模型参数

        参数:
            epoch (int): 当前训练轮次

        返回:
            tensor: 变化检测的预测结果
        """
        self.set_epoch(epoch)  # 更新当前训练轮次
        self.forward_CD()  # 计算前向传播
        self.optimizer_G.zero_grad()  # 清空梯度
        self.backward_G()  # 计算损失并反向传播
        self.optimizer_G.step()  # 更新参数
        return self.change_pred  # 返回变化检测结果

    def save_networks(self, epoch, save_best=False):
        """将所有网络保存到磁盘，覆盖基类方法以支持保存最佳模型

        参数:
            epoch (int/str) -- 当前epoch或'best'等标识
            save_best (bool) -- 是否将模型保存为最佳模型
        """
        for name in self.model_names:
            if isinstance(name, str):
                if save_best:
                    save_filename = 'best_net_%s.pth' % name
                else:
                    save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    # 直接保存GPU上的模型状态字典，避免不必要的GPU-CPU传输
                    torch.save(net.module.state_dict(), save_path)
                else:
                    torch.save(net.state_dict(), save_path)

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

    def get_current_losses(self):
        """返回当前损失的有序字典"""
        losses = {}
        # 只返回loss_names中列出的损失，并且确保该损失确实存在
        for name in self.loss_names:
            loss_name = 'loss_' + name
            if hasattr(self, loss_name):
                loss_value = getattr(self, loss_name)
                # 确保损失值不是None且不是0张量
                if loss_value is not None and not (isinstance(loss_value, torch.Tensor) and loss_value.item() == 0):
                    losses[name] = loss_value.item()
        return losses
