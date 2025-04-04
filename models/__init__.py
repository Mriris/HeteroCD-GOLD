"""此包包含与目标函数、优化和网络架构相关的模块。

要添加名为'dummy'的自定义模型类，您需要添加一个名为'dummy_model.py'的文件，并定义一个继承自BaseModel的子类DummyModel。
您需要实现以下五个函数：
    -- <__init__>:                      初始化类；首先调用BaseModel.__init__(self, opt)。
    -- <set_input>:                     从数据集中解包数据并应用预处理。
    -- <forward>:                       产生中间结果。
    -- <optimize_parameters>:           计算损失，梯度，并更新网络权重。
    -- <modify_commandline_options>:    （可选）添加模型特定选项并设置默认选项。

在<__init__>函数中，您需要定义四个列表：
    -- self.loss_names (str 列表):          指定要绘制和保存的训练损失。
    -- self.model_names (str 列表):         定义训练中使用的网络。
    -- self.visual_names (str 列表):        指定要显示和保存的图像。
    -- self.optimizers (优化器列表):    定义并初始化优化器。您可以为每个网络定义一个优化器。如果两个网络同时更新，可以使用itertools.chain将它们分组。请参考cycle_gan_model.py了解示例。

现在您可以通过指定标志'--model dummy'来使用模型类。
有关更多详细信息，请参阅我们的模板模型类'template_model.py'。
"""

import importlib
from models.base_model import BaseModel


def find_model_using_name(model_name):
    """
    根据给定的模型名称，导入对应的模块。
    
    参数：
        model_name (str) -- 模型名称
    
    返回：
        模块 (python模块) -- 导入的模块
    """
    # 给定模型名称，导入模块 "models/[model_name]_model.py"
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("在 %s.py 中没有找到名为 %s 的模型类" % (model_filename, target_model_name))
        exit(0)
    return model


def get_option_setter(model_name):
    """返回指定模型名称的选项修改器

    参数：
        model_name (str) -- 模型名称 (模块名，例如 "cyclegan")

    返回：
        方法 -- 选项修改器的方法函数
    """
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options


def create_model(opt):
    """
    创建模型
    
    参数：
        opt (Option类) -- 存储所有实验标志的类；应为BaseOptions的子类

    返回：
        模型 (BaseModel类) -- 创建的模型
    """
    model = None
    
    # 根据命令行参数创建模型
    if opt.model == 'gan_cd':
        from .gan_cd import Pix2PixModel
        model = Pix2PixModel(opt)
    elif opt.model == 'HeteGAN':
        from .HeteGAN import Pix2PixModel
        model = Pix2PixModel(opt)
    else:
        raise NotImplementedError('模型[%s]未实现' % opt.model)
        
    # 需要额外配置或初始化，如果有setup方法的话
    if hasattr(model, 'setup'):
        model.setup(opt)
        
    return model
    
    
# 导入供其他模块使用的类和函数
from .DualEUNet import DualEUNet, TripleEUNet
from .HeteGAN import Pix2PixModel
