import numpy as np
import os
import sys
import ntpath
import time
from . import util, html
from subprocess import Popen, PIPE

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """将图像保存到磁盘。

    参数:
        webpage (HTML类) -- 存储这些图像的HTML网页类（更多详细信息请参见html.py）
        visuals (OrderedDict) -- 一个有序字典，存储(名称, 图像(张量或numpy))对
        image_path (str) -- 字符串用于创建图像路径
        aspect_ratio (float) -- 保存的图像的宽高比
        width (int) -- 图像将被调整为宽度 x 宽度

    此函数将把存储在'visuals'中的图像保存到由'webpage'指定的HTML文件中。
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


class Visualizer:
    """此类包含几个可以显示/保存图像和打印/保存日志信息的函数。

    它使用Python库'visdom'进行显示，并使用Python库'dominate'（封装在'HTML'中）创建带有图像的HTML文件。
    """

    def __init__(self, opt):
        """初始化Visualizer类

        参数:
            opt -- 存储所有实验标志；需要是BaseOptions的子类
        步骤1：缓存训练/测试选项
        步骤2：连接到visdom服务器
        步骤3：创建一个HTML对象用于保存HTML过滤器
        步骤4：创建一个日志文件来存储训练损失
        """
        self.opt = opt  # 缓存选项
        self.display_id = getattr(opt, 'display_id', -1)  # 默认禁用visdom
        self.use_html = opt.isTrain
        self.win_size = 512
        self.name = opt.name
        self.port = 8097
        self.saved = False
        if self.display_id > 0:  # 根据<display_port>和<display_server>连接到visdom服务器
            import visdom
            self.ncols = 4
            self.vis = visdom.Visdom(server="http://localhost", port=8097, env="main")
            if not self.vis.check_connection():
                self.create_visdom_connections()

        if self.use_html:  # 在<checkpoints_dir>/web/创建HTML对象；图像将保存在<checkpoints_dir>/web/images/下
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('创建网页目录 %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        # 创建一个日志文件来存储训练损失
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ 训练损失 (%s) ================\n' % now)

    def reset(self):
        """重置self.saved状态"""
        self.saved = False

    def create_visdom_connections(self):
        """如果程序无法连接到Visdom服务器，此函数将在端口<self.port>上启动一个新服务器"""
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\n无法连接到Visdom服务器。\n尝试启动新服务器...')
        print('命令: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result):
        """在visdom上显示当前结果；将当前结果保存到HTML文件。

        参数:
            visuals (OrderedDict) -- 要显示或保存的图像的字典
            epoch (int) -- 当前轮次
            save_result (bool) -- 是否将当前结果保存到HTML文件
        """
        if self.display_id > 0:  # 使用visdom在浏览器中显示图像
            ncols = self.ncols
            if ncols > 0:  # 在一个visdom面板中显示所有图像
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                        table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)  # 创建一个表格css
                # 创建一个图像表格
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image)
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                try:
                    self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                    padding=2, opts=dict(title=title + ' 图像'))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 2,
                                  opts=dict(title=title + ' 标签'))
                except VisdomExceptionBase:
                    self.create_visdom_connections()

            else:  # 在单独的visdom面板中显示每个图像；
                idx = 1
                try:
                    for label, image in visuals.items():
                        image_numpy = util.tensor2im(image)
                        self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                       win=self.display_id + idx)
                        idx += 1
                except VisdomExceptionBase:
                    self.create_visdom_connections()

        if self.use_html and (save_result or not self.saved):  # 如果图像尚未保存，则将它们保存到HTML文件。
            self.saved = True
            # 保存图像到磁盘
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)

            # 更新网站
            webpage = html.HTML(self.web_dir, '实验名称 = %s' % self.name, refresh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('轮次 [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    image_numpy = util.tensor2im(image)
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """在visdom显示上显示当前损失：错误标签和值的字典

        参数:
            epoch (int) -- 当前轮次
            counter_ratio (float) -- 当前轮次的进度（百分比），介于0到1之间
            losses (OrderedDict) -- 以(名称, 浮点数)对的形式存储的训练损失
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' 随时间变化的损失',
                    'legend': self.plot_data['legend'],
                    'xlabel': '轮次',
                    'ylabel': '损失'},
                win=self.display_id)
        except VisdomExceptionBase:
            self.create_visdom_connections()

    # losses: 与plot_current_losses中的|losses|格式相同
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """在控制台上打印当前损失；同时将损失保存到磁盘

        参数:
            epoch (int) -- 当前轮次
            iters (int) -- 当前轮次中的当前训练迭代（在每个轮次结束时重置为0）
            losses (OrderedDict) -- 以(名称, 浮点数)对的形式存储的训练损失
            t_comp (float) -- 每个数据点的计算时间（按批量大小归一化）
            t_data (float) -- 每个数据点的数据加载时间（按批量大小归一化）
        """
        message = '(轮次: %d, 迭代: %d, 耗时: %.3f, 数据加载: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # 打印消息
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # 保存消息
