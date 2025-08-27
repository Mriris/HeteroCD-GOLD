import argparse
import torch


class TestOptions:
    """这个类包含测试选项。"""

    def initialize(self, parser):
        # 基本参数
        parser.add_argument('--dataroot', type=str, default=r'D:\0Program\Datasets\241120\Compare\Datas\Split13', help='图像路径')
        parser.add_argument('--results_dir', type=str, default='./results/', help='测试结果保存路径')
        parser.add_argument('--model_path', type=str, default='./checkpoints8/muagan_dynamic8/best_net_CDcopy.pth', help='指定测试模型路径，例如：checkpoints/GOLD_Test/best_net_CD.pth')
        parser.add_argument('--name', type=str, default='gold_Test3', help='实验名称')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu的id，使用-1表示CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints9', help='模型保存路径')
        parser.add_argument('--init_type', type=str, default='normal',
                            help='网络初始化方式 [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02,
                            help='normal、xavier和orthogonal的缩放因子。')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test等')
        
        # 测试特定参数
        parser.add_argument('--eval', action='store_true', default=True, help='在测试时使用评估模式')
        parser.add_argument('--num_test', type=int, default=float('inf'), help='测试多少图像')
        parser.add_argument('--save_images', action='store_true', default=True, help='是否保存预测结果图像')
        parser.add_argument('--save_comparison', action='store_true', default=True, help='是否生成彩色比对图，用于展示多检漏检情况')
        parser.add_argument('--comparison_dir', type=str, default='comparison', help='彩色比对图保存的子目录名')
        parser.add_argument('--batch_size', type=int, default=1, help='测试批量大小')
        parser.add_argument('--num_workers', type=int, default=4, help='数据加载器的工作线程数')
        parser.add_argument('--load_size', type=int, default=512, help='将图像缩放到此大小')
        parser.add_argument('--crop_size', type=int, default=512, help='然后裁剪到此大小')
        parser.add_argument('--load_t2_opt', action='store_true', default=False, help='是否加载时间点2的光学图像（用于三模态测试）')
        
        # 模型加载参数
        parser.add_argument('--epoch', type=str, default='latest', help='加载哪个epoch的模型（测试时一般不需要）')
        parser.add_argument('--load_iter', type=int, default=0, help='加载特定迭代的模型（测试时一般不需要）')
        parser.add_argument('--continue_train', action='store_true', default=False, help='继续训练（测试时设为False）')
        
        # 模型参数
        parser.add_argument('--verbose', action='store_true', help='如果指定，打印网络架构')
        parser.add_argument('--seed', type=int, default=666, help='随机种子')
        
        # 与训练一致的核心参数（用于蒸馏/权重等）
        parser.add_argument('--use_distill', action='store_true', default=True, help='是否使用蒸馏学习')
        parser.add_argument('--distill_temp', type=float, default=2.0, help='蒸馏学习的温度参数（统一默认2.0）')
        parser.add_argument('--kl_div_reduction', type=str, default='mean', help='KL散度损失的缩减方式 [batchmean | mean | sum | none]')

        # 差异图注意力原子项的初始值（用于生成/评估一致性）
        parser.add_argument('--diff_att_alpha', type=float, default=0.5, help='差异图注意力损失中差异图权重(初始)')
        parser.add_argument('--diff_att_beta', type=float, default=0.3, help='差异图注意力损失中通道注意力权重(初始)')
        parser.add_argument('--diff_att_gamma', type=float, default=0.2, help='差异图注意力损失中空间注意力权重(初始)')
        # parser.add_argument('--diff_att_scale', type=float, default=10.0, help='差异图注意力总损失的缩放因子') # 未使用
        
        # 动态权重分配参数
        parser.add_argument('--use_dynamic_weights', action='store_true', default=True, help='是否使用动态权重分配机制')
        parser.add_argument('--weight_warmup_epochs', type=int, default=20, help='权重热身阶段的轮次数')
        parser.add_argument('--init_cd_weight', type=float, default=1.0, help='变化检测损失的初始权重')
        parser.add_argument('--init_distill_weight', type=float, default=0.3, help='蒸馏损失的初始权重')
        parser.add_argument('--init_diff_att_weight', type=float, default=0.2, help='差异图注意力损失的初始权重')
        parser.add_argument('--init_student_cd_weight', type=float, default=1.0, help='LCD内部学生监督初始权重')
        parser.add_argument('--init_teacher_cd_weight', type=float, default=0.2, help='LCD内部教师监督初始权重')
        parser.add_argument('--init_feat_distill_weight', type=float, default=0.7, help='LDISTILL内部特征蒸馏初始权重')
        parser.add_argument('--init_out_distill_weight', type=float, default=0.3, help='LDISTILL内部输出蒸馏初始权重')
        parser.add_argument('--init_diff_map_weight', type=float, default=0.5, help='LA内部差异图初始权重')
        parser.add_argument('--init_channel_att_weight', type=float, default=0.3, help='LA内部通道注意力初始权重')
        parser.add_argument('--init_spatial_att_weight', type=float, default=0.2, help='LA内部空间注意力初始权重')
        
        # 优化参数
        parser.add_argument('--use_bidirectional_attention', action='store_true', default=True, help='是否使用双向通道注意力机制')
        parser.add_argument('--use_nonlocal_similarity', action='store_true', default=True, help='是否使用非局部相似性匹配模块')
        parser.add_argument('--enhanced_diff_attention', action='store_true', default=True, help='是否使用增强的差异图注意力迁移')
        parser.add_argument('--use_contrastive_loss', action='store_true', default=True, help='是否使用对比学习损失')
        parser.add_argument('--contrastive_temp', type=float, default=0.5, help='对比学习损失的温度参数')
        parser.add_argument('--contrastive_weight', type=float, default=10.0, help='对比学习损失的权重')
        parser.add_argument('--feature_fusion_type', type=str, default='concat', choices=['concat', 'add', 'attention'], help='特征融合类型')
        
        # 学习率参数
        parser.add_argument('--lr', type=float, default=0.0005, help='初始学习率')
        parser.add_argument('--lr_policy', type=str, default='cosine', help='学习率策略')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='学习率衰减间隔')
        parser.add_argument('--beta1', type=float, default=0.5, help='Adam优化器的beta1参数')
        
        # 轻量化模型参数
        parser.add_argument('--use_lightweight', action='store_true', default=False, help='是否使用轻量化模型')
        parser.add_argument('--channel_reduction', type=float, default=0.5, help='通道数减少比例')
        parser.add_argument('--attention_reduction_ratio', type=int, default=32, help='注意力模块的reduction ratio')
        
        # 混合精度和梯度裁剪
        parser.add_argument('--use_amp', action='store_true', default=False, help='是否使用混合精度训练')
        parser.add_argument('--gradient_clip_norm', type=float, default=1.0, help='梯度裁剪的最大范数值')
        parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='梯度累积步数')
        
        # CE 类权重与蒸馏特征掩码权重（用于验证阶段一致性）
        parser.add_argument('--ce_weight_bg', type=float, default=0.1, help='CE背景类权重')
        parser.add_argument('--ce_weight_fg', type=float, default=0.9, help='CE前景类权重')
        parser.add_argument('--feature_mask_pos_weight', type=float, default=8.0, help='蒸馏特征掩码正样本权重')
        parser.add_argument('--feature_mask_neg_weight', type=float, default=0.2, help='蒸馏特征掩码负样本权重')
        parser.add_argument('--teacher_entropy_weight', type=float, default=0.0, help='教师熵正则权重(默认关闭)')
        
        self.isTrain = False
        return parser

    def parse(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)
        opt = parser.parse_args()
        opt.isTrain = self.isTrain  # 测试或训练

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
