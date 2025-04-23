import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks


class BaseModel(ABC):
    """这个类是所有模型的抽象基类。
    要创建一个子类，需要实现以下五个函数:
        -- <__init__>:                      初始化网络，定义网络，初始化损失函数和优化器
        -- <set_input>:                     从数据集中解析数据并应用预处理
        -- <forward>:                       生成中间结果
        -- <optimize_parameters>:           计算损失，梯度并更新网络权重
        -- <modify_commandline_options>:    （可选）添加模型特定的选项并设置默认选项。

    在BaseModel类中定义的实用函数:
        -- <setup>:                         （由<__init__>调用）打印网络，创建调度器
        -- <parallelize>:                   在多GPU上并行化网络
        -- <data_dependent_initialize>:     （可选）在第一个数据批次上初始化网络
        -- <forward>:                       用于测试的推断步骤
        -- <get_image_paths>:               获取图像路径以保存输出图像
        -- <update_learning_rate>:          更新所有网络的学习率；由<optimize_parameters>在每个迭代中调用
        -- <get_current_visuals>:           获取待显示的图像
        -- <get_current_losses>:            获取待显示的损失
        -- <save_networks>:                 保存所有网络到磁盘
        -- <load_networks>:                 从磁盘加载所有网络
        -- <print_networks>:                打印网络的总数
        -- <set_requires_grad>:             避免梯度计算的方便函数
    """

    def __init__(self, opt, is_train=True):
        """初始化BaseModel类.

        参数:
            opt (Option类)-- 存储所有实验标志；需要是BaseOptions的子类

        当创建自定义类时，需要实现自己的初始化。
        在这个函数中，应该首先调用 <BaseModel.__init__(self, opt)>
        然后定义在测试和训练模式下需要使用的网络，损失函数，优化器。
        此外，也可以使用<setup>函数初始化模型。
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # 获取设备名称: CPU 或 GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # 保存checkpoints的目录
        
        # 检查是否存在preprocess属性，如果不存在则设置默认值
        if not hasattr(opt, 'preprocess'):
            opt.preprocess = ''
            
        if opt.preprocess != 'scale_width':  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # 用于学习率调整策略，在训练期间跟踪的最佳验证指标

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """添加新的模型特定选项，并重写现有选项的默认值。

        参数:
            parser          -- 原始选项解析器
            is_train (bool) -- 是否训练阶段或测试阶段。可以使用此标志添加特定于训练或测试的选项。

        返回:
            修改后的解析器。
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """从数据加载器解包输入数据并执行必要的预处理步骤。

        参数:
            input (dict): 包括数据本身及其元数据信息。
        """
        pass

    @abstractmethod
    def forward(self):
        """运行前向传递；由<optimize_parameters>和<test>函数调用。"""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """计算损失、梯度并更新网络权重；在每次训练迭代中调用"""
        pass

    def setup(self, opt):
        """执行模型所有的设置，打印网络，创建调度器等。
        这个函数必须在每次初始化时调用
        """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        
        # 当继续训练时，加载保存的模型
        if not self.isTrain or opt.continue_train:
            # 确定要加载的模型文件
            if opt.continue_train and hasattr(opt, 'epoch') and opt.epoch != 'latest':
                # 使用指定的epoch
                load_suffix = opt.epoch
                if isinstance(load_suffix, str) and not load_suffix.endswith('.pth'):
                    print(f"断点续训：尝试加载epoch {load_suffix}的模型")
                else:
                    print(f"断点续训：尝试加载指定的模型文件 {load_suffix}")
            else:
                # 默认使用最新的epoch或指定的迭代
                load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
                print(f"尝试加载模型：{load_suffix}")
            
            # 设置加载错误后的回退策略
            try:
                # 先尝试加载指定的模型
                self.load_networks(load_suffix)
            except Exception as e:
                # 如果加载失败，尝试回退到其他模型
                print(f"加载指定模型失败: {e}")
                
                # 如果是继续训练，优先回退到best或latest模型
                if opt.continue_train:
                    try:
                        print("尝试加载最佳模型...")
                        self.load_networks('best')
                    except Exception as e2:
                        print(f"加载最佳模型失败: {e2}")
                        try:
                            print("尝试加载最新模型...")
                            self.load_networks('latest')
                        except Exception as e3:
                            print(f"加载最新模型也失败: {e3}")
                            print("警告：无法加载任何模型，将使用初始化的权重")
                else:
                    print("非断点续训模式，将使用初始化的权重")
        
        # 打印网络结构
        self.print_networks(opt.verbose)

    def parallelize(self):
        """用torch.nn.DataParallel在多GPU上并行化模型"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                setattr(self, 'net' + name, torch.nn.DataParallel(net, self.opt.gpu_ids))

    def data_dependent_initialize(self, data):
        """在数据上初始化模型

        可选地，训练模型可能需要其他准备步骤（例如，预计算风格表示）。
        """
        pass

    def eval(self):
        """使网络进入评估模式"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        """测试期间使用前向传播"""
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """计算额外的输出图像以供可视化"""
        pass

    def get_image_paths(self):
        """ 返回图像路径，用于保存/展示结果 """
        return self.image_paths

    def optimize_parameters(self):
        """计算损失，梯度，并更新网络权重；在训练期间每个迭代中调用"""
        pass

    def update_learning_rate(self):
        """更新所有网络的学习率；在训练期间每个迭代中调用"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('学习率 %.7f -> %.7f' % (old_lr, lr))

    def get_current_visuals(self):
        """返回待显示的图像"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """返回待显示的训练损失"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) 可以将1x1 Tensor变为float
        return errors_ret

    def save_networks(self, epoch, save_best=False):
        """将所有网络保存到磁盘

        参数:
            epoch (int) -- 当前epoch; 用于在文件名中添加 'epoch_'
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
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """修复InstanceNorm checkpoints不兼容性（在multi-GPU训练中）

        这不需要在一般场景中使用，但是需要用于继续旧的训练任务。
        """
        key = keys[i]
        if i + 1 == len(keys):  # 在键的末尾
            if module.affine:
                state_dict[key] = module.bias.data
                state_dict[key.replace('bias', 'weight')] = module.weight.data
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        """从磁盘加载所有网络

        参数:
            epoch (int) -- 当前epoch; 用于在文件名中添加 'epoch_'
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                
                # 检查文件是否存在，如果不存在则给出警告
                if not os.path.exists(load_path):
                    print(f"警告: 模型文件 {load_path} 不存在！")
                    
                    # 如果是latest，尝试寻找最新的模型文件
                    if epoch == 'latest' or epoch == 'best':
                        print(f"尝试寻找可用的模型文件...")
                        # 收集所有适用的模型文件
                        model_files = [f for f in os.listdir(self.save_dir) 
                                       if f.endswith(f'_net_{name}.pth')]
                        
                        if model_files:
                            # 如果文件名形式为"数字_net_XX.pth"，则按数字排序，取最大的
                            model_files.sort(key=lambda x: int(x.split('_')[0]) 
                                            if x.split('_')[0].isdigit() else -1, 
                                            reverse=True)
                            
                            load_filename = model_files[0]
                            load_path = os.path.join(self.save_dir, load_filename)
                            print(f"找到模型文件: {load_path}")
                        else:
                            print(f"未找到任何可用的模型文件！")
                            continue
                    else:
                        # 如果是特定epoch且不存在，则继续下一个模型
                        continue
                
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('从%s加载网络' % load_path)
                
                try:
                    # 加载模型，处理可能的设备不匹配
                    state_dict = torch.load(load_path, map_location=str(self.device))
                    
                    if hasattr(state_dict, '_metadata'):
                        del state_dict._metadata

                    # 尝试移除可能存在的"module."前缀
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        if k.startswith('module.'):
                            new_key = k[7:]  # 移除 'module.' 前缀
                        else:
                            new_key = k
                        new_state_dict[new_key] = v
                    
                    # 加载状态字典
                    net.load_state_dict(new_state_dict)
                    print(f"成功加载模型 {name}")
                except Exception as e:
                    print(f"加载模型时出错: {e}")
                    print(f"尝试继续处理...")

    def print_networks(self, verbose):
        """打印网络的总参数数量

        参数:
            verbose (bool) -- 如果verbose: 打印网络的结构
        """
        print('---------- 网络初始化 -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[网络 %s] 总参数数量 : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """在更新网络权重时设置需要或不需要梯度

        参数:
            nets (network list)   -- 网络列表
            requires_grad (bool)  -- 是否需要梯度
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def generate_visuals_for_evaluation(self, data, mode):
        """在评估模式下计算额外的输出图像，用于保存而不是显示"""
        pass
