import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import os
from scipy.ndimage import gaussian_filter1d

#=================== 参数设置区域 ===================
# 热身轮次
WARMUP_EPOCHS = 20
# 图表平滑参数
SMOOTHING_SIGMA = 1  # 高斯滤波sigma参数，值越大曲线越平滑，仅PPT美化
# 图表大小和样式参数
FIGURE_WIDTH = 12
FIGURE_HEIGHT = 6
LEGEND_FONTSIZE = 19
TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 16
#=================== 参数设置结束 ===================

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取loss_log.txt文件
file_path = 'checkpoints/muagan_dynamic8/loss_log.txt'

# 存储数据的列表
data = []

# 读取文件并解析数据
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        # 跳过非数据行
        if not line.strip() or '================' in line:
            continue
        
        # 提取轮次和权重数据
        epoch_match = re.search(r'轮次: (\d+)', line)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            
            # 提取各项损失值
            cd_match = re.search(r'CD: ([\d\.]+)', line)
            distill_match = re.search(r'Distill: ([\d\.]+)', line)
            diff_att_match = re.search(r'Diff_Att: ([\d\.]+)', line)
            teacher_match = re.search(r'Teacher: ([\d\.]+)', line)
            dynamic_weight_match = re.search(r'Dynamic_Weight: ([\d\.]+)', line)
            
            if all([cd_match, distill_match, diff_att_match, teacher_match, dynamic_weight_match]):
                cd = float(cd_match.group(1))
                distill = float(distill_match.group(1))
                diff_att = float(diff_att_match.group(1))
                teacher = float(teacher_match.group(1))
                dynamic_weight = float(dynamic_weight_match.group(1))
                
                data.append({
                    'epoch': epoch,
                    'CD': cd,
                    'Distill': distill,
                    'Diff_Att': diff_att,
                    'Teacher': teacher,
                    'Dynamic_Weight': dynamic_weight
                })

# 转换为DataFrame
df = pd.DataFrame(data)

# 确保数据按轮次排序
df = df.sort_values('epoch')

# 计算每个轮次的平均权重
epoch_avg = df.groupby('epoch').mean().reset_index()

# 确保存储目录存在
os.makedirs('doc/image/动态权重分配', exist_ok=True)

# 获取所有轮次范围
min_epoch = epoch_avg['epoch'].min()
max_epoch = epoch_avg['epoch'].max()

# 计算每个权重相对于自身的变化趋势
log_columns = ['CD', 'Distill', 'Diff_Att', 'Teacher', 'Dynamic_Weight']

# 先过滤出热身轮次后的数据
post_warmup_df = epoch_avg[epoch_avg['epoch'] >= WARMUP_EPOCHS].copy()

# 检查是否有连续的数据，输出验证信息
print(f"原始数据总数: {len(epoch_avg)}")
print(f"热身后数据总数: {len(post_warmup_df)}")
print(f"热身后数据轮次范围: {post_warmup_df['epoch'].min()} - {post_warmup_df['epoch'].max()}")

# 创建归一化数据框
normalized_df = pd.DataFrame()
normalized_df['epoch'] = post_warmup_df['epoch']

# 只对热身轮次后的数据进行自身归一化处理
for col in log_columns:
    # Min-Max归一化，使每列的值都在0-1之间
    min_val = post_warmup_df[col].min()
    max_val = post_warmup_df[col].max()
    normalized_df[col] = (post_warmup_df[col] - min_val) / (max_val - min_val)

# 应用平滑处理
smoothed_df = normalized_df.copy()
for col in log_columns:
    # 使用高斯滤波平滑数据，sigma控制平滑程度
    smoothed_df[col] = gaussian_filter1d(normalized_df[col], sigma=SMOOTHING_SIGMA)
    
    # 平滑后对数据再次缩放，确保最大值仍为1
    if smoothed_df[col].max() > 0:  # 避免除以0
        smoothed_df[col] = smoothed_df[col] / smoothed_df[col].max()

# 绘制每个权重相对于自身的变化趋势
plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

# 绘制热身轮次后的归一化数据
for col in log_columns:
    label_map = {
        'CD': '变化检测损失权重',
        'Distill': '蒸馏损失权重',
        'Diff_Att': '差异图注意力损失权重',
        'Teacher': '教师网络损失权重',
        'Dynamic_Weight': '动态权重损失'
    }
    # 使用平滑后的数据绘图
    plt.plot(smoothed_df['epoch'], smoothed_df[col], label=label_map[col], linewidth=2)

# 设置x轴范围，包括热身期
plt.xlim(min_epoch, max_epoch)

# 设置y轴范围为0到1
plt.ylim(0, 1.05)

# 标记热身区域
plt.axvspan(min_epoch, WARMUP_EPOCHS, alpha=0.2, color='gray')
plt.axvline(x=WARMUP_EPOCHS, linestyle='--', color='red', alpha=0.7)
plt.text(WARMUP_EPOCHS+1, 0.95, f'热身', color='red', fontsize=12)

# 设置图表标题和标签
# plt.title('各损失权重自身相对变化趋势 (0-1归一化)', fontsize=TITLE_FONTSIZE)
plt.xlabel('训练轮次', fontsize=LABEL_FONTSIZE)
plt.ylabel('权重自身相对值', fontsize=LABEL_FONTSIZE)
plt.legend(fontsize=LEGEND_FONTSIZE, bbox_to_anchor=(1.02, 0.4), loc='center right')
plt.grid(True, alpha=0.3)
# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig('doc/image/动态权重分配/weight_self_normalized.png', dpi=300, bbox_inches='tight')

print("分析完成，权重变化图已保存至doc/image/动态权重分配/目录")

# # 显示图表
# plt.show() 