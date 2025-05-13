import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import cv2
from matplotlib.patches import Arrow, FancyArrowPatch
from matplotlib import colors
import matplotlib.font_manager as fm

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']  # 优先使用的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
plt.rcParams['font.family'] = 'sans-serif'  # 使用sans-serif字体

# 加载图像
img_a = np.array(Image.open('../../datasets/sample/A.png'))
img_b = np.array(Image.open('../../datasets/sample/B.png'))
img_c = np.array(Image.open('../../datasets/sample/C.png'))
img_label = np.array(Image.open('../../datasets/sample/Label.png'))

# 创建注意力热力图(示例) - 基于标签图
attention_map = np.zeros((img_label.shape[0], img_label.shape[1]), dtype=np.float32)
for i in range(img_label.shape[0]):
    for j in range(img_label.shape[1]):
        # 如果是变化区域，增加注意力权重
        if img_label[i, j, 0] > 200:  # 假设Label图像中白色区域表示变化
            attention_map[max(0, i-10):min(i+10, attention_map.shape[0]), 
                          max(0, j-10):min(j+10, attention_map.shape[1])] += 1.0

# 归一化注意力图
attention_map = cv2.GaussianBlur(attention_map, (21, 21), 0)
attention_map = attention_map / np.max(attention_map)

# 计算热力图 - 保存为全局变量，以便后面单独保存
heatmap_img = img_b.copy()  # 使用图像B作为基础
for i in range(3):
    weight = attention_map * 0.7
    heatmap_img[:,:,i] = np.clip(
        img_b[:,:,i] * (1 - weight) + 
        weight * 255 * ([0,1,1][i]), 
        0, 255).astype(np.uint8)

# 准备绘图 - 修改为3:10比例的横向布局
# 假设高度为3，宽度为10
height = 3
width = 10
plt.figure(figsize=(width, height))

# 创建1x4的网格
gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1])

# 设置标题和全局风格
# plt.suptitle("差异图注意力迁移机制", fontsize=14, fontweight='bold')
plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0.02, right=0.98, top=0.85, bottom=0.02)

# 四个子图横向排列
ax1 = plt.subplot(gs[0, 0])
ax1.imshow(img_a[:,:,:3])
ax1.set_title("事前光学", fontsize=10)
ax1.axis('off')

ax2 = plt.subplot(gs[0, 1])
ax2.imshow(img_c[:,:,:3])
ax2.set_title("事后光学", fontsize=10)
ax2.axis('off')

ax3 = plt.subplot(gs[0, 2])
# 使用标签图作为差异图
ax3.imshow(img_label[:,:,:3])
ax3.set_title("差异图", fontsize=10)
ax3.axis('off')

# 绘制热力图
ax4 = plt.subplot(gs[0, 3])
# 直接使用前面计算好的热力图
ax4.imshow(heatmap_img[:,:,:3])
ax4.set_title("注意力热力图(事后SAR)", fontsize=10)
ax4.axis('off')

# # 添加注意力迁移的箭头
# arrow = FancyArrowPatch(
#     (0.25, 0.5), (0.75, 0.5),
#     transform=plt.gcf().transFigure,
#     connectionstyle="arc3,rad=.2",
#     arrowstyle="fancy,head_width=10,head_length=10",
#     color="red",
#     linewidth=3
# )
# plt.gcf().add_artist(arrow)

# # 添加解释性文字
# plt.figtext(0.5, 0.02, 
#            "差异图注意力迁移机制：通过识别时相图像之间的差异区域，\n引导模型注意力聚焦在发生变化的关键区域，提高变化检测的准确性。", 
#            ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

# 保存图片
plt.savefig('attention_mechanism_figure.png', dpi=300, bbox_inches='tight')
plt.savefig('attention_mechanism_figure.pdf', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()

# 单独保存热力图，不包含标题和其他元素
plt.figure(figsize=(5, 5))
plt.imshow(heatmap_img[:,:,:3])
plt.axis('off')
plt.tight_layout()
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 去除所有边距
plt.savefig('heatmap_only.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.savefig('heatmap_only.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close() 