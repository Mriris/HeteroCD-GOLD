from .base_model import BaseModel
import math
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .TripleEUNet import DualEUNet, TripleEUNet, LightweightTripleEUNet
from .base_model import BaseModel
from .loss import *
from .loss import HeterogeneousAttentionDistillationLoss, DifferenceAttentionLoss


class TripleHeteCD(BaseModel):
    def __init__(self, opt, is_train=True):
        """初始化

        参数:
            opt (Option类) -- 存储所有实验标志的类；需要是BaseOptions的子类
        """
        BaseModel.__init__(self, opt, is_train=True)

        # 是否使用三分支网络和蒸馏学习
        self.use_distill = opt.use_distill
        # 是否使用轻量化模型
        self.use_lightweight = getattr(opt, 'use_lightweight', False)

        # 添加动态权重分配相关参数
        self.use_dynamic_weights = opt.use_dynamic_weights  # 直接从opt中获取参数值
        self.weight_warmup_epochs = opt.weight_warmup_epochs  # 权重热身阶段的轮次数
        self.current_epoch = 0  # 当前训练轮次

        # 初始权重设置（任务级）
        self.init_cd_weight = opt.init_cd_weight
        self.init_distill_weight = opt.init_distill_weight
        self.init_diff_att_weight = opt.init_diff_att_weight

        # LCD 内部（学生/教师）初始权重
        self.init_student_cd_weight = getattr(opt, 'init_student_cd_weight', 100.0)
        self.init_teacher_cd_weight = getattr(opt, 'init_teacher_cd_weight', 20.0)
        # LDISTILL 内部（特征/输出）初始权重
        self.init_feat_distill_weight = getattr(opt, 'init_feat_distill_weight', 0.7)
        self.init_out_distill_weight = getattr(opt, 'init_out_distill_weight', 0.3)
        # LA 内部（差异图/通道/空间）初始权重
        self.init_diff_map_weight = getattr(opt, 'init_diff_map_weight', 0.5)
        self.init_channel_att_weight = getattr(opt, 'init_channel_att_weight', 0.3)
        self.init_spatial_att_weight = getattr(opt, 'init_spatial_att_weight', 0.2)

        # CE 与 Dice 在 LCD 内部的固定比例（避免使用变化比例动态项）
        self.ce_in_lcd_weight = getattr(opt, 'ce_in_lcd_weight', 100.0)
        self.dice_in_lcd_weight = getattr(opt, 'dice_in_lcd_weight', 150.0)

        # 交叉熵类别权重与蒸馏特征掩码权重
        self.ce_weight_bg = getattr(opt, 'ce_weight_bg', 0.1)
        self.ce_weight_fg = getattr(opt, 'ce_weight_fg', 0.9)
        self.feature_mask_pos_weight = getattr(opt, 'feature_mask_pos_weight', 8.0)
        self.feature_mask_neg_weight = getattr(opt, 'feature_mask_neg_weight', 0.2)

        # 教师熵正则（可选，默认关闭）
        self.teacher_entropy_weight = getattr(opt, 'teacher_entropy_weight', 0.0)

        # 指定要打印的训练损失。训练/测试脚本将调用<BaseModel.get_current_losses>
        self.loss_names = ['CD']
        if self.use_distill:
            self.loss_names.extend(['Distill', 'Diff_Att'])
        # 分层不确定性权重参数
        if self.use_dynamic_weights:
            # 任务级：LCD / LDISTILL / LA
            self.log_vars_task = nn.Parameter(torch.zeros(3))
            # LCD 内部：学生CD / 教师CD
            self.log_vars_cd = nn.Parameter(torch.zeros(2))
            # LDISTILL 内部：特征蒸馏 / 输出蒸馏
            self.log_vars_distill = nn.Parameter(torch.zeros(2))
            # LA 内部：差异图 / 通道注意力 / 空间注意力
            self.log_vars_att = nn.Parameter(torch.zeros(3))
        else:
            self.log_vars_task = None
            self.log_vars_cd = None
            self.log_vars_distill = None
            self.log_vars_att = None

        # 指定要保存/显示的图像。
        self.change_pred = None
        self.teacher_pred = None
        self.isTrain = is_train
        # 指定要保存到磁盘的模型。
        self.model_names = ['CD']

        # 定义网络
        if self.use_distill:
            if self.use_lightweight:
                # 使用轻量化三分支网络
                print("使用轻量化模型")
                self.netCD = LightweightTripleEUNet(
                    3, 2, 
                    channel_reduction=getattr(opt, 'channel_reduction', 0.5),
                    attention_reduction_ratio=getattr(opt, 'attention_reduction_ratio', 32)
                )
            else:
                # 使用标准三分支网络
                print("使用标准模型")
                self.netCD = TripleEUNet(3, 2)
                
            # 使用蒸馏损失（仅返回特征/输出两部分）
            self.distill_loss = HeterogeneousAttentionDistillationLoss(
                temperature=getattr(opt, 'distill_temp', 2.0),
                reduction=getattr(opt, 'kl_div_reduction', 'batchmean')
            )
            # 差异图注意力迁移损失（仅用于计算三个原子项；总和在外部用不确定性权重融合）
            self.diff_att_loss = DifferenceAttentionLoss(
                reduction='mean',
                alpha=getattr(opt, 'diff_att_alpha', 0.5),
                beta=getattr(opt, 'diff_att_beta', 0.3),
                gamma=getattr(opt, 'diff_att_gamma', 0.2)
            )
        else:
            self.netCD = DualEUNet(3, 2)

        self.netCD.to(opt.gpu_ids[0])
        self.is_train = is_train

        if is_train:
            self.netCD = torch.nn.DataParallel(self.netCD, opt.gpu_ids)  # 多GPU支持

        if self.isTrain:
            # 将模型与log_vars添加到优化器中
            params = [
                {'params': filter(lambda p: p.requires_grad, self.netCD.parameters())},
            ]
            if self.use_dynamic_weights:
                params.append({'params': self.log_vars_task, 'lr': opt.lr * 0.1})
                params.append({'params': self.log_vars_cd, 'lr': opt.lr * 0.1})
                params.append({'params': self.log_vars_distill, 'lr': opt.lr * 0.1})
                params.append({'params': self.log_vars_att, 'lr': opt.lr * 0.1})
            self.optimizer_G = torch.optim.AdamW(params, lr=opt.lr,
                                                 betas=(0.9, 0.999), weight_decay=0.01)
            self.optimizers.append(self.optimizer_G)

    def set_epoch(self, epoch):
        """设置当前训练轮次，用于动态权重计算

        参数:
            epoch (int): 当前训练轮次
        """
        self.current_epoch = epoch

    def _compute_group_weights(self, log_vars, init_weights_tensor):
        """基于不确定性的分组权重计算，支持warmup与按初始量级缩放"""
        if not self.use_dynamic_weights or log_vars is None:
            return init_weights_tensor
        
        # 确保log_vars与init_weights_tensor在同一设备上
        log_vars = log_vars.to(init_weights_tensor.device)
        precision = torch.nn.functional.softplus(-log_vars) + 1e-8
        
        if self.current_epoch < self.weight_warmup_epochs:
            progress = self.current_epoch / max(1, self.weight_warmup_epochs)
            alpha = 0.5 * (1 - math.cos(progress * math.pi))
            fixed = init_weights_tensor / (init_weights_tensor.sum() + 1e-8)
            dynamic = precision / (precision.sum() + 1e-8)
            weights = (1 - alpha) * fixed + alpha * dynamic
        else:
            weights = precision / (precision.sum() + 1e-8)
        # 将权重缩放回初始量级（去除二次逐元素缩放）
        weights = weights * (init_weights_tensor.sum() + 1e-8)
        return weights

    def get_group_weights(self):
        """返回四组权重：任务级、LCD内部、LDISTILL内部、LA内部"""
        device = self.netCD.module.parameters().__next__().device if isinstance(self.netCD, torch.nn.DataParallel) else next(self.netCD.parameters()).device
        task_init = torch.tensor([self.init_cd_weight, self.init_distill_weight, self.init_diff_att_weight], device=device)
        cd_init = torch.tensor([self.init_student_cd_weight, self.init_teacher_cd_weight], device=device)
        distill_init = torch.tensor([self.init_feat_distill_weight, self.init_out_distill_weight], device=device)
        att_init = torch.tensor([self.init_diff_map_weight, self.init_channel_att_weight, self.init_spatial_att_weight], device=device)
        task_w = self._compute_group_weights(self.log_vars_task, task_init)
        cd_w = self._compute_group_weights(self.log_vars_cd, cd_init)
        distill_w = self._compute_group_weights(self.log_vars_distill, distill_init)
        att_w = self._compute_group_weights(self.log_vars_att, att_init)
        # 任务级权重 clip 与 LCD 保底
        total = task_w.sum() + 1e-8
        min_lcd = 0.6
        max_distill = 0.3
        max_att = 0.2
        target = torch.tensor([min_lcd, max_distill, max_att], device=task_w.device) * total
        task_w = torch.stack([
            torch.max(task_w[0], target[0]),
            torch.min(task_w[1], target[1]),
            torch.min(task_w[2], target[2]),
        ])
        task_w = task_w / (task_w.sum() + 1e-8) * total
        return task_w, cd_w, distill_w, att_w

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
            cls_weights = torch.tensor([self.ce_weight_bg, self.ce_weight_fg]).cuda()
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
            cls_weights = torch.tensor([self.ce_weight_bg, self.ce_weight_fg]).cuda()
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
        # 类权重
        cls_weights = torch.tensor([self.ce_weight_bg, self.ce_weight_fg]).cuda()
        self.label = self.label.long()

        # 主要变化检测损失（学生）
        ce_loss = CE_Loss(self.change_pred, self.label, cls_weights=cls_weights)
        dice_loss = Dice_loss(self.change_pred, self.label)
        student_cd_loss = ce_loss * self.ce_in_lcd_weight + dice_loss * self.dice_in_lcd_weight

        # 初始化蒸馏与注意力损失
        self.loss_Distill = torch.tensor(0.0).cuda()
        self.loss_Diff_Att = torch.tensor(0.0).cuda()

        # 教师监督（合并入 LCD 内部）
        teacher_cd_loss = torch.tensor(0.0).cuda()
        if self.use_distill and hasattr(self, 'teacher_out') and self.teacher_out is not None:
            teacher_out_resized = F.interpolate(
                self.teacher_out,
                size=(self.change_pred.size(2), self.change_pred.size(3)),
                mode='bilinear',
                align_corners=True
            )
            teacher_ce_loss = CE_Loss(teacher_out_resized, self.label, cls_weights=cls_weights)
            teacher_dice_loss = Dice_loss(teacher_out_resized, self.label)
            teacher_cd_loss = teacher_ce_loss * self.ce_in_lcd_weight + teacher_dice_loss * self.dice_in_lcd_weight
            if self.teacher_entropy_weight > 0.0:
                teacher_probs = F.softmax(teacher_out_resized, dim=1)
                teacher_entropy = -torch.sum(teacher_probs * torch.log(teacher_probs + 1e-6), dim=1).mean()
                teacher_cd_loss = teacher_cd_loss + self.teacher_entropy_weight * teacher_entropy

            # 构造蒸馏用的特征掩码
            if len(self.label.shape) == 3:
                label_mask = self.label.unsqueeze(1)
            else:
                label_mask = self.label
            feature_mask = torch.zeros_like(label_mask, dtype=torch.float)
            feature_mask[label_mask == 1] = self.feature_mask_pos_weight
            feature_mask[label_mask == 0] = self.feature_mask_neg_weight
            feature_mask = F.interpolate(
                feature_mask,
                size=self.student_feat.size()[2:],
                mode='nearest'
            )

            # 差异图注意力三个原子项
            diff_att_total, diff_att_loss, channel_att_loss, spatial_att_loss = self.diff_att_loss(
                self.student_feat, self.teacher_feat,
                self.opt_t1_feat, self.opt_t2_feat, self.sar_t2_feat
            )

            # 蒸馏两个子项（特征/输出）
            feat_loss, out_loss = self.distill_loss(
                self.student_feat,
                self.teacher_feat,
                self.change_pred,
                teacher_out_resized,
                self.opt_t1_feat,
                self.opt_t2_feat,
                self.sar_t2_feat,
                feature_mask
            )

            # 计算分层不确定性权重
            task_w, cd_w, distill_w, att_w = self.get_group_weights()

            # 蒸馏与教师监督热身/渐入
            if self.weight_warmup_epochs and self.weight_warmup_epochs > 0:
                progress = min(max(self.current_epoch / float(self.weight_warmup_epochs), 0.0), 1.0)
                distill_alpha = 0.5 * (1 - math.cos(progress * math.pi))
            else:
                distill_alpha = 1.0

            # 组合 LCD（学生/教师）
            self.loss_CD = student_cd_loss * cd_w[0] + (teacher_cd_loss * distill_alpha) * cd_w[1]
            # 组合 LDISTILL（特征/输出）
            self.loss_Distill = distill_alpha * (feat_loss * distill_w[0] + out_loss * distill_w[1])
            # 组合 LA（差异图/通道/空间）
            self.loss_Diff_Att = (diff_att_loss * att_w[0] +
                                   channel_att_loss * att_w[1] +
                                   spatial_att_loss * att_w[2])

            # 任务级融合
            self.loss_G = (self.loss_CD * task_w[0] +
                           self.loss_Distill * task_w[1] +
                           self.loss_Diff_Att * task_w[2])
        else:
            # 不使用蒸馏时，仅有学生 LCD
            task_w, cd_w, distill_w, att_w = self.get_group_weights()
            self.loss_CD = student_cd_loss * cd_w[0]  # 仅学生项
            self.loss_G = self.loss_CD * task_w[0]
            
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
                                print(f"未找到任何可用的模型文件。")

                # 最终加载模型
                if os.path.exists(load_path):
                    net = getattr(self, 'net' + name)
                    state_dict = torch.load(load_path)
                    if isinstance(net, torch.nn.DataParallel):
                        net.module.load_state_dict(state_dict)
                    else:
                        net.load_state_dict(state_dict)
                    print(f"模型已加载: {load_path}")

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
        
        # 添加教师网络损失以便监控
        if hasattr(self, 'teacher_loss'):
            losses['Teacher'] = self.teacher_loss.item()
            
        return losses
