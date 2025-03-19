import dominate
from dominate.tags import meta, h3, table, tr, td, p, a, img, br
import os


class HTML:
    """此类用于在可视化期间保存图像及网页信息。

    该类使用dominate包来创建HTML文件。它保存实验期间生成的图像到HTML文件中。
    """

    def __init__(self, web_dir, title, refresh=0):
        """初始化HTML类

        参数:
            web_dir (str) -- 存放网站的文件夹
            title (str)   -- 网站的名称
            refresh (int) -- 刷新网站的间隔（0：不刷新）
        """
        self.title = title
        self.web_dir = web_dir
        self.img_dir = os.path.join(self.web_dir, 'images')
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        self.doc = dominate.document(title=title)
        if refresh > 0:
            with self.doc.head:
                meta(http_equiv="refresh", content=str(refresh))

    def get_image_dir(self):
        """返回图像文件夹的路径"""
        return self.img_dir

    def add_header(self, text):
        """在HTML文件中添加标题

        参数:
            text (str) -- 标题文本
        """
        with self.doc:
            h3(text)

    def add_images(self, ims, txts, links, width=400):
        """添加图像到HTML文件

        参数:
            ims (str list)   -- 图像的路径列表
            txts (str list)  -- 图像的标题列表
            links (str list) -- 图像的超链接列表
            width (int)      -- 图像宽度
        """
        self.t = table(border=1, style="table-layout: fixed;")  # 创建表格
        self.doc.add(self.t)
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=os.path.join('images', link)):
                                img(style="width:%dpx" % width, src=os.path.join('images', im))
                            br()
                            p(txt)

    def save(self):
        """保存当前内容到HTML文件"""
        html_file = '%s/index.html' % self.web_dir
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()


if __name__ == '__main__':  # 我们在本地进行简单的测试。
    html = HTML('web/', 'test_html')
    html.add_header('hello world')

    ims, txts, links = [], [], []
    for n in range(4):
        ims.append('image_%d.png' % n)
        txts.append('text_%d' % n)
        links.append('image_%d.png' % n)
    html.add_images(ims, txts, links)
    html.save()
