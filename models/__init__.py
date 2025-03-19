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
    """导入模块"models/[model_name]_model.py"。

    在文件中，将实例化名为[ModelName]Model()的类。它必须是BaseModel的子类，
    并且不区分大小写。
    """
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
                and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("在 %s.py 中找不到名为 %s 的模型" % (model_filename, target_model_name))
        exit(0)

    return model


def get_option_setter(model_name):
    """返回模型类的静态方法<modify_commandline_options>。"""
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options


def create_model(opt):
    """根据选项创建模型。

    此函数包装了CustomDatasetDataLoader类。
    这是此包与'train.py'/'test.py'之间的主要接口

    示例:
        >>> from models import create_model
        >>> model = create_model(opt)
    """
    model = find_model_using_name(opt.model)
    instance = model(opt)
    print("已创建模型 [%s]" % type(instance).__name__)
    return instance
