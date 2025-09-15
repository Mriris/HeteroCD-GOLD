import torch
import argparse


def str2bool(v):
    """将字符串转换为布尔值"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('布尔值期望为 yes/no, true/false, t/f, y/n, 1/0')


class TrainOptions:
    """这个类包含训练选项。

    它还包括在BaseOptions中定义的共享选项。
    """

    def initialize(self, parser):
        # 基本参数
        parser.add_argument('--dataroot', default=r'/data/jingwei/yantingxuan/Datasets/CityCN/Split24',
                            help='图像路径')
        parser.add_argument('--name', type=str, default='gold18',
                            help='实验名称。决定了在哪里存储样本和模型')
        parser.add_argument('--gpu_ids', type=str, default='0',
                            help='gpu的id：例如 0  0,1,2, 0,2。使用-1表示CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints9', help='模型保存路径')
        parser.add_argument('--init_type', type=str, default='normal',
                            help='网络初始化方式 [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02,
                            help='normal、xavier和orthogonal的缩放因子。')
        parser.add_argument('--use_amp', action='store_true', default=False, 
                            help='是否使用混合精度训练（需要PyTorch>=1.6），可以加速训练，但可能会降低稳定性')
        parser.add_argument('--gradient_clip_norm', type=float, default=0.5,
                            help='梯度裁剪的最大范数值，用于避免梯度爆炸（调整为0.5增强稳定性）')
        # 附加参数
        parser.add_argument('--epoch', type=str, default='latest',
                            help='加载哪个epoch？设置为latest使用最新的缓存模型')
        parser.add_argument('--verbose', action='store_true', help='如果指定，打印网络架构')
        parser.add_argument('--continue_train', action='store_true', help='继续训练：从--epoch加载网络')
        parser.add_argument('--load_iter', type=int, default=0, help='迭代从哪个加载？如果设为0，则从最近的epoch加载')

        # visdom和HTML可视化参数
        parser.add_argument('--display_id', type=int, default=-1, help='visdom显示窗口id，设为-1禁用visdom')
        parser.add_argument('--print_freq', type=int, default=100,
                            help='在控制台上显示训练结果的频率')
        # 网络保存和加载参数
        parser.add_argument('--phase', type=str, default='train', help='train, val, test等')
        # 训练参数
        parser.add_argument('--load_size', type=int, default=512, help='将图像缩放到此大小')
        parser.add_argument('--crop_size', type=int, default=512, help='然后裁剪到此大小')
        parser.add_argument('--n_epochs', type=int, default=400, help='使用初始学习率的epoch数量')
        parser.add_argument('--beta1', type=float, default=0.5, help='adam的动量项')
        parser.add_argument('--lr', type=float, default=0.0005, help='adam的初始学习率')
        parser.add_argument('--lr_policy', type=str, default='cosine',
                            help='学习率策略。[linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50,
                            help='每lr_decay_iters次迭代乘以一个gamma')
        parser.add_argument('--batch_size', type=int, default=8, help='输入批量大小')
        parser.add_argument('--num_workers', type=int, default=8, help='数据加载器的工作线程数')
        parser.add_argument('--seed', type=int, default=666, help='随机种子')
        
        # 蒸馏学习参数
        parser.add_argument('--use_distill', type=str2bool, default=True, help='是否使用蒸馏学习')
        parser.add_argument('--distill_temp', type=float, default=2.0, help='蒸馏学习的温度参数（统一默认2.0）')
        parser.add_argument('--kl_div_reduction', type=str, default='mean', 
                            help='KL散度损失的缩减方式 [batchmean | mean | sum | none]；推荐mean')
        
        # 差异图注意力迁移参数（仅用于原子项，外部不确定性权重融合）
        parser.add_argument('--diff_att_alpha', type=float, default=0.6, help='差异图注意力损失中差异图权重(初始)')
        parser.add_argument('--diff_att_beta', type=float, default=0.25, help='差异图注意力损失中通道注意力权重(初始)')
        parser.add_argument('--diff_att_gamma', type=float, default=0.15, help='差异图注意力损失中空间注意力权重(初始)')
        
        # 动态权重分配参数
        parser.add_argument('--use_dynamic_weights', type=str2bool, default=True, help='是否使用动态权重分配机制')
        parser.add_argument('--weight_warmup_epochs', type=int, default=20, help='权重热身阶段的轮次数')
        # 任务级初始权重（缩放到1量级，避免总损失量级过大）
        parser.add_argument('--init_cd_weight', type=float, default=1.0, help='变化检测损失的初始权重')
        parser.add_argument('--init_distill_weight', type=float, default=0.3, help='蒸馏损失的初始权重（下调以防早期主导）')
        parser.add_argument('--init_diff_att_weight', type=float, default=0.3, help='差异图注意力损失的初始权重')
        # LCD 内部初始权重（保持学生:教师=5:1，但缩小到1量级）
        parser.add_argument('--init_student_cd_weight', type=float, default=1.0, help='LCD内部学生监督初始权重')
        parser.add_argument('--init_teacher_cd_weight', type=float, default=0.4, help='LCD内部教师监督初始权重（提升，强化教师监督稳定性）')
        # LDISTILL 内部初始权重（维持0.7/0.3比例）
        parser.add_argument('--init_feat_distill_weight', type=float, default=0.7, help='LDISTILL内部特征蒸馏初始权重')
        parser.add_argument('--init_out_distill_weight', type=float, default=0.3, help='LDISTILL内部输出蒸馏初始权重')
        # LA 内部初始权重（维持0.5/0.3/0.2比例）
        parser.add_argument('--init_diff_map_weight', type=float, default=0.5, help='LA内部差异图初始权重')
        parser.add_argument('--init_channel_att_weight', type=float, default=0.3, help='LA内部通道注意力初始权重')
        parser.add_argument('--init_spatial_att_weight', type=float, default=0.2, help='LA内部空间注意力初始权重')

        # LCD 内部CE/Dice组合系数（缩放到1量级，保留约2:3的相对关系）
        parser.add_argument('--ce_in_lcd_weight', type=float, default=1.0, help='LCD内部CE损失系数')
        parser.add_argument('--dice_in_lcd_weight', type=float, default=1.5, help='LCD内部Dice损失系数')

        # CE 类权重与蒸馏特征掩码权重（重新平衡以改善精准率）
        parser.add_argument('--ce_weight_bg', type=float, default=0.1, help='CE背景类权重')
        parser.add_argument('--ce_weight_fg', type=float, default=0.9, help='CE前景类权重')
        parser.add_argument('--feature_mask_pos_weight', type=float, default=4.0, help='蒸馏特征掩码正样本权重')
        parser.add_argument('--feature_mask_neg_weight', type=float, default=0.5, help='蒸馏特征掩码负样本权重')
        parser.add_argument('--teacher_entropy_weight', type=float, default=0.0, help='教师熵正则权重(默认关闭)')
        
        # 轻量化模型参数
        parser.add_argument('--use_lightweight', type=str2bool, default=False, help='是否使用轻量化模型')
        parser.add_argument('--channel_reduction', type=float, default=0.5, help='通道数减少比例')
        parser.add_argument('--attention_reduction_ratio', type=int, default=32, help='注意力模块的reduction ratio')

        # 训练时在线数据增强
        parser.add_argument('--aug_in_train', type=str2bool, default=True, help='是否在训练阶段启用在线数据增强')
        parser.add_argument('--aug_rotate_prob', type=float, default=0.5, help='随机旋转概率')
        parser.add_argument('--aug_rotate_degree', type=float, default=180, help='随机旋转角度范围（±度）')
        parser.add_argument('--aug_hflip_prob', type=float, default=0.5, help='水平翻转概率')
        parser.add_argument('--aug_vflip_prob', type=float, default=0.5, help='垂直翻转概率')
        parser.add_argument('--aug_exchange_time_prob', type=float, default=0.0, help='时间交换概率（A/B交换）')
        parser.add_argument('--aug_use_photometric', type=str2bool, default=True, help='是否启用光照与颜色扰动')
        parser.add_argument('--aug_brightness_delta', type=float, default=10, help='亮度随机偏移范围（±）')
        parser.add_argument('--aug_contrast_range', type=float, nargs=2, default=(0.8, 1.2), help='对比度缩放范围')
        parser.add_argument('--aug_saturation_range', type=float, nargs=2, default=(0.8, 1.2), help='饱和度缩放范围')
        parser.add_argument('--aug_hue_delta', type=float, default=10, help='色调偏移范围（±，单位：度）')
        parser.add_argument('--aug_cat_max_ratio', type=float, default=0.75, help='随机裁剪时单类最大占比阈值，不满足则重试')
        parser.add_argument('--aug_use_scale_random_crop', type=str2bool, default=True, help='是否启用尺度随机裁剪')
        parser.add_argument('--aug_use_random_blur', type=str2bool, default=True, help='是否启用随机高斯模糊')
        
        self.isTrain = True
        return parser

    def parse(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)
        opt = parser.parse_args()
        opt.isTrain = self.isTrain  # 训练或测试

        # 设置gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
