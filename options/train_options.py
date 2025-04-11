import torch
import argparse


class TrainOptions():
    """这个类包含训练选项。

    它还包括在BaseOptions中定义的共享选项。
    """

    def initialize(self, parser):
        # 基本参数
        parser.add_argument('--dataroot', default='/data/jingwei/yantingxuan/Datasets/CityCN/Split8',
                            help='图像路径')
        parser.add_argument('--name', type=str, default='muagan_dynamic5',
                            help='实验名称。决定了在哪里存储样本和模型')
        parser.add_argument('--gpu_ids', type=str, default='1,2',
                            help='gpu的id：例如 0  0,1,2, 0,2。使用-1表示CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints8', help='模型保存路径')
        parser.add_argument('--init_type', type=str, default='normal',
                            help='网络初始化方式 [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02,
                            help='normal、xavier和orthogonal的缩放因子。')
        # 附加参数
        parser.add_argument('--epoch', type=str, default='latest',
                            help='加载哪个epoch？设置为latest使用最新的缓存模型')
        parser.add_argument('--verbose', action='store_true', help='如果指定，打印网络架构')
        parser.add_argument('--continue_train', action='store_true', help='继续训练：从--epoch加载网络')
        parser.add_argument('--load_iter', type=int, default=0, help='迭代从哪个加载？如果设为0，则从最近的epoch加载')

        # visdom和HTML可视化参数
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
        parser.add_argument('--batch_size', type=int, default=4, help='输入批量大小')
        parser.add_argument('--num_workers', type=int, default=8, help='数据加载器的工作线程数')
        parser.add_argument('--seed', type=int, default=666, help='随机种子')
        
        # 蒸馏学习参数
        parser.add_argument('--use_distill', action='store_true', default=True, help='是否使用蒸馏学习')
        parser.add_argument('--distill_temp', type=float, default=2.0, help='蒸馏学习的温度参数')
        parser.add_argument('--distill_alpha', type=float, default=0.3, help='特征蒸馏损失权重')
        parser.add_argument('--distill_beta', type=float, default=0.4, help='输出蒸馏损失权重')
        parser.add_argument('--distill_gamma', type=float, default=0.3, help='差异图注意力迁移损失权重')
        parser.add_argument('--kl_div_reduction', type=str, default='mean', 
                            help='KL散度损失的缩减方式 [mean | batchmean | sum | none]。推荐使用batchmean，与KL散度数学定义一致，仅按批次大小平均。')
        
        # 差异图注意力迁移参数
        parser.add_argument('--diff_att_alpha', type=float, default=0.5, help='差异图注意力损失中差异图权重')
        parser.add_argument('--diff_att_beta', type=float, default=0.3, help='差异图注意力损失中通道注意力权重')
        parser.add_argument('--diff_att_gamma', type=float, default=0.2, help='差异图注意力损失中空间注意力权重')
        parser.add_argument('--diff_att_scale', type=float, default=10.0, help='差异图注意力总损失的缩放因子')
        
        # 动态权重分配参数
        parser.add_argument('--use_dynamic_weights', action='store_true', default=True, help='是否使用动态权重分配机制')
        parser.add_argument('--weight_warmup_epochs', type=int, default=20, help='权重热身阶段的轮次数')
        parser.add_argument('--init_cd_weight', type=float, default=120.0, help='变化检测损失的初始权重')
        parser.add_argument('--init_distill_weight', type=float, default=5.0, help='蒸馏损失的初始权重')
        parser.add_argument('--init_diff_att_weight', type=float, default=15.0, help='差异图注意力损失的初始权重')
        
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
