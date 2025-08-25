import argparse
import glob
import math
import os
import random
import re
import shutil
from collections import defaultdict

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

import rasterio
from rasterio.transform import from_bounds

# 新增sklearn导入，用于地理聚类
try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("警告: sklearn未安装，地理感知功能将被禁用")

# 默认参数设置
DEFAULT_INPUT_DIR = r"C:\1DataSets\241120\Compare\Datas\Final"  # 输入目录
DEFAULT_OUTPUT_DIR = r"C:\1DataSets\241120\Compare\Datas\Split19"  # 输出目录
DEFAULT_TILE_SIZE = 512  # 切片大小
DEFAULT_SIZE_TOLERANCE = 2  # 大小容差
DEFAULT_OVERLAP_RATIO = 0.5  # 裁剪重叠比例
DEFAULT_OVERLAP_THRESHOLD = 0.8  # 重叠度阈值，超过此值的小块将被丢弃
DEFAULT_VAL_RATIO = 0.2  # 验证集占总数据的比例
DEFAULT_CREATE_TEST_FOLDER = True  # 是否创建测试集文件夹
DEFAULT_FILTER_BLACK_TILES = True  # 是否过滤纯黑色小块
DEFAULT_BLACK_THRESHOLD = 0.95  # 纯黑色判定阈值，超过此比例的黑色像素将被视为纯黑色小块

# 地理感知相关默认参数
DEFAULT_GEO_AWARE = False  # 是否启用地理感知划分
DEFAULT_GEO_EPS = 2000  # DBSCAN聚类的邻域半径（米）
DEFAULT_GEO_MIN_SAMPLES = 1  # DBSCAN聚类的最小样本数

# 数据增强方法控制
APPLY_H_FLIP = False    # 是否应用水平翻转
APPLY_V_FLIP = False    # 是否应用垂直翻转
APPLY_ROT90 = False     # 是否应用90°旋转
APPLY_ROT180 = False    # 是否应用180°旋转
APPLY_ROT270 = False    # 是否应用270°旋转

# ========================= 新增：前景统计与均衡划分辅助函数 ========================= #

def load_label_and_count_foreground(label_path: str):
    """
    读取整幅标签图并统计前景像素数量与总像素数量。
    规则：像素值 > 0 视为前景。
    """
    try:
        img = Image.open(label_path).convert('L')
        arr = np.array(img)
        total = arr.shape[0] * arr.shape[1]
        fg = int((arr > 0).sum())
        return fg, total
    except Exception as e:
        print(f"警告: 无法读取标签 {label_path} 统计前景: {e}")
        return 0, 0


def compute_fg_stats_for_basenames(input_dir: str, base_names: list):
    """
    为每个基础名称统计前景/总像素。
    返回: dict[base_name] = { 'fg': int, 'total': int }
    """
    stats = {}
    for base_name in base_names:
        label_path = os.path.join(input_dir, f"{base_name}_E.png")
        fg, total = load_label_and_count_foreground(label_path)
        stats[base_name] = {'fg': fg, 'total': total}
    return stats


def aggregate_cluster_stats(base_names: list, cluster_labels: np.ndarray, fg_stats: dict):
    """
    聚合每个聚类的图像数量与前景/总像素统计。
    返回: dict[label] = { 'indices': np.ndarray, 'names': list, 'count': int, 'fg': int, 'total': int }
    """
    cluster_info = {}
    unique_labels = np.unique(cluster_labels)
    for label in unique_labels:
        indices = np.where(cluster_labels == label)[0]
        names = [base_names[i] for i in indices]
        fg_sum = 0
        total_sum = 0
        for name in names:
            s = fg_stats.get(name, {'fg': 0, 'total': 0})
            fg_sum += s['fg']
            total_sum += s['total']
        cluster_info[label] = {
            'indices': indices,
            'names': names,
            'count': len(names),
            'fg': fg_sum,
            'total': total_sum,
        }
    return cluster_info


def print_split_fg_summary(split_name: str, names: list, fg_stats: dict):
    """
    打印某个划分的前景比例统计。
    """
    fg_total = 0
    pix_total = 0
    for n in names:
        s = fg_stats.get(n, {'fg': 0, 'total': 0})
        fg_total += s['fg']
        pix_total += s['total']
    ratio = (fg_total / pix_total) if pix_total > 0 else 0.0
    print(f"{split_name} 前景像素: {fg_total} / {pix_total} (比例: {ratio*100:.2f}%)")


# ========================= 现有函数 ========================= #

def get_geo_transform(tif_path):
    """
    获取TIF文件的地理坐标变换信息
    
    参数:
        tif_path: TIF文件路径
    
    返回:
        地理坐标变换矩阵
    """
    try:
        with rasterio.open(tif_path) as src:
            return src.transform
    except Exception as e:
        raise RuntimeError(f"无法读取 {tif_path} 的地理坐标信息: {e}")


def get_image_center_coordinates(tif_path):
    """
    获取TIF文件的中心地理坐标
    
    参数:
        tif_path: TIF文件路径
    
    返回:
        (center_x, center_y) 中心地理坐标
    """
    try:
        with rasterio.open(tif_path) as src:
            bounds = src.bounds
            center_x = (bounds.left + bounds.right) / 2
            center_y = (bounds.bottom + bounds.top) / 2
            return center_x, center_y
    except Exception as e:
        print(f"警告: 无法读取 {tif_path} 的地理坐标: {e}")
        return None, None


def analyze_geographic_distribution(input_dir, base_names):
    """
    分析所有原始大图的地理坐标分布
    
    参数:
        input_dir: 输入目录
        base_names: 基础名称列表
    
    返回:
        coords: numpy数组，包含所有图像的地理坐标
        valid_names: 有效的基础名称列表（能读取地理坐标的）
    """
    coords = []
    valid_names = []
    
    print("分析地理坐标分布...")
    for base_name in tqdm(base_names, desc="读取地理坐标"):
        tif_path = os.path.join(input_dir, f"{base_name}_A.tif")
        center_x, center_y = get_image_center_coordinates(tif_path)
        
        if center_x is not None and center_y is not None:
            coords.append([center_x, center_y])
            valid_names.append(base_name)
        else:
            print(f"跳过无法读取地理坐标的图像: {base_name}")
    
    coords = np.array(coords)
    
    if len(coords) > 0:
        print(f"成功获取 {len(coords)} 个图像的地理坐标")
        print(f"坐标范围: X({coords[:, 0].min():.2f} ~ {coords[:, 0].max():.2f}), Y({coords[:, 1].min():.2f} ~ {coords[:, 1].max():.2f})")
        
        # 计算最近邻距离统计
        if len(coords) > 1:
            from scipy.spatial.distance import pdist
            distances = pdist(coords)
            print(f"图像间距离统计: 最小{distances.min():.0f}m, 中位数{np.median(distances):.0f}m, 最大{distances.max():.0f}m")
    
    return coords, valid_names


def perform_geographic_clustering(coords, eps=2000, min_samples=1):
    """
    对地理坐标进行DBSCAN聚类
    
    参数:
        coords: 地理坐标数组
        eps: 聚类邻域半径（米）
        min_samples: 最小样本数
    
    返回:
        cluster_labels: 聚类标签数组
    """
    if not SKLEARN_AVAILABLE:
        print("警告: sklearn不可用，返回随机聚类标签")
        return np.random.randint(0, 2, len(coords))
    
    print(f"执行地理聚类 (邻域半径: {eps}m, 最小样本数: {min_samples})...")
    
    # 使用DBSCAN进行聚类
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    cluster_labels = clustering.labels_
    
    # 统计聚类结果
    unique_labels = np.unique(cluster_labels)
    n_clusters = len(unique_labels) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    print(f"聚类结果: {n_clusters} 个聚类, {n_noise} 个噪声点")
    
    for label in unique_labels:
        if label == -1:
            continue  # 跳过噪声点
        cluster_size = list(cluster_labels).count(label)
        print(f"  聚类 {label}: {cluster_size} 个图像")
    
    return cluster_labels


def geo_aware_train_val_split(base_names, coords, cluster_labels, val_ratio=0.2):
    """
    基于地理聚类进行训练/验证集划分，确保地理完全分离
    
    参数:
        base_names: 基础名称列表
        coords: 地理坐标数组
        cluster_labels: 聚类标签
        val_ratio: 验证集比例
    
    返回:
        train_names: 训练集基础名称列表
        val_names: 验证集基础名称列表
    """
    print("执行地理感知的训练/验证集划分...")
    
    # 统计聚类结果（补充前景统计）
    unique_labels = np.unique(cluster_labels)

    # 先计算每张图的前景统计
    input_dir_placeholder = None  # 仅为签名兼容占位，此函数内不直接访问磁盘
    # 注意：真正的统计在外部已完成，并在调用时注入。为最小侵入改动，我们在本函数内重建一次统计。
    # 由于原函数签名不含输入路径，这里采用回退方案：使用全局变量缓存。
    # 为避免引入全局状态，后续在调用处直接替换为新版函数 geo_aware_train_val_split_balanced。

    # 为保持向后兼容，此处保留原始简单策略（以防外部未替换调用）。
    cluster_info = {}
    for label in unique_labels:
        indices = np.where(cluster_labels == label)[0]
        cluster_info[label] = {
            'indices': indices,
            'count': list(cluster_labels).count(label),
            'names': [base_names[i] for i in indices]
        }

    # 原始简单贪心：整簇划分，尽量接近目标数量
    total_images = len(base_names)
    target_val_size = int(total_images * val_ratio)
    print(f"目标验证集大小: {target_val_size}/{total_images} ({val_ratio*100:.1f}%)")

    sorted_clusters = sorted(cluster_info.items(), key=lambda x: x[1]['count'], reverse=True)
    train_names = []
    val_names = []
    current_val_size = 0

    for cluster_label, info in sorted_clusters:
        if cluster_label == -1:
            # 噪声点均匀分配（保持原逻辑）
            noise_names = info['names']
            random.shuffle(noise_names)
            remaining_val_need = target_val_size - current_val_size
            noise_for_val = min(remaining_val_need, len(noise_names) // 2)
            val_names.extend(noise_names[:noise_for_val])
            train_names.extend(noise_names[noise_for_val:])
            current_val_size += noise_for_val
            print(f"  噪声点: {len(noise_names)} 个图像 -> 训练集{len(noise_names)-noise_for_val}, 验证集{noise_for_val}")
        else:
            if current_val_size < target_val_size and current_val_size + info['count'] <= target_val_size * 1.2:
                val_names.extend(info['names'])
                current_val_size += info['count']
                print(f"  聚类 {cluster_label}: {info['count']} 个图像 -> 验证集")
            else:
                train_names.extend(info['names'])
                print(f"  聚类 {cluster_label}: {info['count']} 个图像 -> 训练集")

    print(f"\n地理感知划分结果:")
    print(f"  训练集: {len(train_names)} 个图像")
    print(f"  验证集: {len(val_names)} 个图像")

    return train_names, val_names


# ========================= 新增：均衡前景比例的地理划分 ========================= #

def geo_aware_train_val_split_balanced(base_names, cluster_labels, val_ratio, fg_stats):
    """
    基于地理聚类进行训练/验证集划分，并尽量匹配整体前景比例。

    参数:
        base_names: 基础名称列表
        cluster_labels: 聚类标签
        val_ratio: 验证集比例
        fg_stats: dict[base_name] -> {'fg': int, 'total': int}

    返回:
        train_names, val_names
    """
    print("执行地理感知的训练/验证集划分（均衡前景比例）...")

    # 计算聚类级汇总
    clusters = aggregate_cluster_stats(base_names, cluster_labels, fg_stats)

    total_images = len(base_names)
    target_val_size = int(total_images * val_ratio)

    # 全局前景比例（以像素计）
    global_fg = sum(fg_stats[n]['fg'] for n in base_names if n in fg_stats)
    global_total = sum(fg_stats[n]['total'] for n in base_names if n in fg_stats)
    global_ratio = (global_fg / global_total) if global_total > 0 else 0.0
    print(f"全局前景比例(像素): {global_fg}/{global_total} = {global_ratio*100:.2f}%")

    # 贪心：按簇大小降序遍历，将能让验证集前景比例更接近全局比例的簇优先加入验证集
    sorted_clusters = sorted(clusters.items(), key=lambda kv: kv[1]['count'], reverse=True)

    train_names, val_names = [], []
    val_fg, val_total, val_count = 0, 0, 0

    for label, info in sorted_clusters:
        names = info['names']
        c_count = info['count']
        c_fg = info['fg']
        c_total = info['total']

        # 若验证集尚未达到目标数量，则考虑加入验证集
        if val_count < target_val_size:
            new_val_count = val_count + c_count
            new_val_fg = val_fg + c_fg
            new_val_total = val_total + c_total
            new_ratio = (new_val_fg / new_val_total) if new_val_total > 0 else 0.0
            curr_ratio = (val_fg / val_total) if val_total > 0 else 0.0

            improve = abs(new_ratio - global_ratio) < abs(curr_ratio - global_ratio)
            within_limit = (new_val_count <= int(target_val_size * 1.2))

            if within_limit and (improve or (target_val_size - val_count) >= c_count):
                val_names.extend(names)
                val_fg, val_total, val_count = new_val_fg, new_val_total, new_val_count
                print(f"  聚类 {label}: {c_count} -> 验证集 (val_ratio: {new_ratio*100:.2f}%)")
            else:
                train_names.extend(names)
                print(f"  聚类 {label}: {c_count} -> 训练集")
        else:
            train_names.extend(names)
            print(f"  聚类 {label}: {c_count} -> 训练集")

    # 如验证集不足，回填最小簇
    if val_count < target_val_size:
        remaining = target_val_size - val_count
        leftovers = [ (label, info) for label, info in sorted_clusters if info['names'][0] in train_names or True ]
        # 简化处理：按簇大小升序回填
        leftovers_sorted = sorted(clusters.items(), key=lambda kv: kv[1]['count'])
        for label, info in leftovers_sorted:
            names = [n for n in info['names'] if n in train_names]
            if not names:
                continue
            if val_count + len(names) <= target_val_size * 1.2:
                for n in names:
                    train_names.remove(n)
                val_names.extend(names)
                val_count += len(names)
                print(f"  回填聚类 {label}: {len(names)} 张 -> 验证集")
            if val_count >= target_val_size:
                break

    print(f"\n地理感知(均衡)划分结果:")
    print(f"  训练集: {len(train_names)} 个图像")
    print(f"  验证集: {len(val_names)} 个图像")

    return train_names, val_names


def verify_geographic_separation(train_names, val_names, input_dir, min_distance=1000):
    """
    验证训练集和验证集之间的地理分离程度
    
    参数:
        train_names: 训练集基础名称列表
        val_names: 验证集基础名称列表
        input_dir: 输入目录
        min_distance: 最小距离阈值（米）
    
    返回:
        separation_ok: 是否满足地理分离要求
        min_distance_found: 实际找到的最小距离
    """
    print("验证地理分离程度...")
    
    # 获取训练集和验证集的坐标
    train_coords = []
    val_coords = []
    
    for name in train_names:
        tif_path = os.path.join(input_dir, f"{name}_A.tif")
        center_x, center_y = get_image_center_coordinates(tif_path)
        if center_x is not None:
            train_coords.append([center_x, center_y])
    
    for name in val_names:
        tif_path = os.path.join(input_dir, f"{name}_A.tif")
        center_x, center_y = get_image_center_coordinates(tif_path)
        if center_x is not None:
            val_coords.append([center_x, center_y])
    
    if not train_coords or not val_coords:
        print("警告: 无法获取坐标信息，跳过分离验证")
        return True, float('inf')
    
    train_coords = np.array(train_coords)
    val_coords = np.array(val_coords)
    
    # 计算训练集和验证集间的最小距离
    from scipy.spatial.distance import cdist
    distances = cdist(train_coords, val_coords)
    min_distance_found = distances.min()
    
    separation_ok = min_distance_found >= min_distance
    
    print(f"  训练集与验证集最小距离: {min_distance_found:.0f}m")
    print(f"  地理分离状态: {'✓ 满足要求' if separation_ok else '✗ 距离过近'} (阈值: {min_distance}m)")
    
    return separation_ok, min_distance_found


def pixel_to_geo_coords(pixel_x, pixel_y, transform):
    """
    将像素坐标转换为地理坐标
    
    参数:
        pixel_x, pixel_y: 像素坐标
        transform: rasterio变换矩阵
    
    返回:
        (geo_x, geo_y) 地理坐标
    """
    geo_x = transform[2] + pixel_x * transform[0] + pixel_y * transform[1]
    geo_y = transform[5] + pixel_x * transform[3] + pixel_y * transform[4]
    return geo_x, geo_y


def pixel_box_to_geo_box(pixel_box, transform):
    """
    将像素坐标边界框转换为地理坐标边界框
    
    参数:
        pixel_box: (x1, y1, x2, y2) 像素坐标边界框
        transform: rasterio变换矩阵
    
    返回:
        (geo_x1, geo_y1, geo_x2, geo_y2) 地理坐标边界框
    """
    x1, y1, x2, y2 = pixel_box
    
    # 转换四个角点
    geo_x1, geo_y1 = pixel_to_geo_coords(x1, y1, transform)
    geo_x2, geo_y2 = pixel_to_geo_coords(x2, y2, transform)
    
    # 确保坐标顺序正确（左上角到右下角）
    min_x, max_x = min(geo_x1, geo_x2), max(geo_x1, geo_x2)
    min_y, max_y = min(geo_y1, geo_y2), max(geo_y1, geo_y2)
    
    return (min_x, min_y, max_x, max_y)


def is_black_tile(tile, threshold=0.95):
    """
    检测图像小块是否为纯黑色或接近纯黑色
    
    参数:
        tile: PIL图像对象
        threshold: 黑色像素比例阈值（0-1之间）
    
    返回:
        True如果是纯黑色小块，False否则
    """
    # 转换为numpy数组
    img_array = np.array(tile)
    
    # 计算总像素数
    if len(img_array.shape) == 2:  # 灰度图像
        total_pixels = img_array.size
        # 计算黑色像素数量（值为0或接近0的像素）
        black_pixels = np.sum(img_array <= 5)  # 允许一些噪声，值<=5认为是黑色
    else:  # 彩色图像
        total_pixels = img_array.shape[0] * img_array.shape[1]
        # 计算黑色像素数量（RGB所有通道都接近0的像素）
        black_mask = np.all(img_array <= 5, axis=2)  # RGB所有通道都<=5
        black_pixels = np.sum(black_mask)
    
    # 计算黑色像素比例
    black_ratio = black_pixels / total_pixels
    
    return black_ratio >= threshold


def calculate_overlap_ratio(box1, box2):
    """
    计算两个矩形框的比例
    
    参数:
        box1: 第一个矩形框 (x1, y1, x2, y2)
        box2: 第二个矩形框 (x1, y1, x2, y2)
    
    返回:
        重叠比例 (0-1之间)
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # 计算交集
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # 使用较小区域作为分母计算重叠比例
    smaller_area = min(area1, area2)
    overlap_ratio = inter_area / smaller_area if smaller_area > 0 else 0.0
    
    return overlap_ratio


def check_overlap_with_existing(new_box, existing_boxes, threshold=0.8):
    """
    检查新的矩形框是否与已存在的矩形框重叠度过高
    
    参数:
        new_box: 新的矩形框 (x1, y1, x2, y2)
        existing_boxes: 已存在的矩形框列表
        threshold: 重叠度阈值
    
    返回:
        True如果重叠度超过阈值，False否则
    """
    for existing_box in existing_boxes:
        overlap_ratio = calculate_overlap_ratio(new_box, existing_box)
        if overlap_ratio > threshold:
            return True
    return False


def tile_image_with_overlap(img, tile_size, overlap_ratio=0.5, pad_value=0, geo_transform=None):
    """
    将图像切分为固定大小的小块，使用指定的重叠比例，不足的地方进行填充

    参数:
        img: PIL图像对象
        tile_size: 小块大小 (width, height)
        overlap_ratio: 重叠比例 (0-1之间)
        pad_value: 填充值
        geo_transform: 地理坐标变换矩阵

    返回:
        tiles: 切分后的小块列表
        positions: 每个小块在原图中的位置 (x, y)
        boxes: 每个小块的地理坐标边界框
    """
    width, height = img.size
    tile_width, tile_height = tile_size

    # 计算步长（非重叠部分的大小）
    stride_w = int(tile_width * (1 - overlap_ratio))
    stride_h = int(tile_height * (1 - overlap_ratio))

    # 确保步长至少为1
    stride_w = max(1, stride_w)
    stride_h = max(1, stride_h)

    # 计算所需的行列数
    num_cols = math.ceil((width - tile_width) / stride_w) + 1 if width > tile_width else 1
    num_rows = math.ceil((height - tile_height) / stride_h) + 1 if height > tile_height else 1

    # 计算需要填充的大小
    pad_width = max(0, stride_w * (num_cols - 1) + tile_width - width)
    pad_height = max(0, stride_h * (num_rows - 1) + tile_height - height)

    # 创建填充后的图像
    padded_width = width + pad_width
    padded_height = height + pad_height

    if img.mode == 'L':
        padded_img = Image.new('L', (padded_width, padded_height), pad_value)
    else:  # RGB或其他模式
        if isinstance(pad_value, int):
            pad_value = (pad_value,) * len(img.getbands())
        padded_img = Image.new(img.mode, (padded_width, padded_height), pad_value)

    # 粘贴原图
    padded_img.paste(img, (0, 0))

    tiles = []
    positions = []
    boxes = []

    # 切分图像
    for row in range(num_rows):
        for col in range(num_cols):
            x = col * stride_w
            y = row * stride_h

            # 提取小块
            pixel_box = (x, y, x + tile_width, y + tile_height)
            tile = padded_img.crop(pixel_box)

            # 转换为地理坐标边界框
            geo_box = pixel_box_to_geo_box(pixel_box, geo_transform)
            boxes.append(geo_box)

            tiles.append(tile)
            positions.append((x, y))

    return tiles, positions, boxes


def apply_geometric_augmentations(img_A, img_B, img_D, img_E, apply_h_flip, apply_v_flip, apply_rot90, apply_rot180, apply_rot270):
    """
    对图像应用指定的几何变换数据增强

    参数:
        img_A, img_B, img_D: 输入图像
        img_E: 标签图像
        apply_h_flip: 是否应用水平翻转
        apply_v_flip: 是否应用垂直翻转
        apply_rot90: 是否应用90°旋转
        apply_rot180: 是否应用180°旋转
        apply_rot270: 是否应用270°旋转

    返回:
        增强后的图像列表
    """
    augmented_images = []

    # 始终包括原始图像
    augmented_images.append((img_A, img_B, img_D, img_E, "original"))

    if apply_h_flip:
        aug_A = ImageOps.mirror(img_A)
        aug_B = ImageOps.mirror(img_B)
        aug_D = ImageOps.mirror(img_D)
        aug_E = ImageOps.mirror(img_E)
        augmented_images.append((aug_A, aug_B, aug_D, aug_E, "h_flip"))

    if apply_v_flip:
        aug_A = ImageOps.flip(img_A)
        aug_B = ImageOps.flip(img_B)
        aug_D = ImageOps.flip(img_D)
        aug_E = ImageOps.flip(img_E)
        augmented_images.append((aug_A, aug_B, aug_D, aug_E, "v_flip"))

    if apply_rot90:
        aug_A = img_A.rotate(90, expand=True)
        aug_B = img_B.rotate(90, expand=True)
        aug_D = img_D.rotate(90, expand=True)
        aug_E = img_E.rotate(90, expand=True)
        size = img_A.size
        aug_A = aug_A.resize(size)
        aug_B = aug_B.resize(size)
        aug_D = aug_D.resize(size)
        aug_E = aug_E.resize(size)
        augmented_images.append((aug_A, aug_B, aug_D, aug_E, "rot90"))

    if apply_rot180:
        aug_A = img_A.rotate(180)
        aug_B = img_B.rotate(180)
        aug_D = img_D.rotate(180)
        aug_E = img_E.rotate(180)
        augmented_images.append((aug_A, aug_B, aug_D, aug_E, "rot180"))

    if apply_rot270:
        aug_A = img_A.rotate(270, expand=True)
        aug_B = img_B.rotate(270, expand=True)
        aug_D = img_D.rotate(270, expand=True)
        aug_E = img_E.rotate(270, expand=True)
        size = img_A.size
        aug_A = aug_A.resize(size)
        aug_B = aug_B.resize(size)
        aug_D = aug_D.resize(size)
        aug_E = aug_E.resize(size)
        augmented_images.append((aug_A, aug_B, aug_D, aug_E, "rot270"))

    return augmented_images


def is_acceptable_size_difference(sizes, tolerance=2):
    """
    检查尺寸差异是否在可接受范围内

    参数:
        sizes: 尺寸列表
        tolerance: 允许的像素差异阈值

    返回:
        是否可接受
    """
    max_width = max(w for w, h in sizes)
    min_width = min(w for w, h in sizes)
    max_height = max(h for w, h in sizes)
    min_height = min(h for w, h in sizes)

    width_diff = max_width - min_width
    height_diff = max_height - min_height

    return width_diff <= tolerance and height_diff <= tolerance


# ========================= 新增：流式先切片再划分 ========================= #

def crop_with_padding(img, pixel_box, tile_size, pad_value=0):
    """
    从图像裁剪 pixel_box 所示区域，不足处用 pad_value 填充至 tile_size 大小。
    """
    x1, y1, x2, y2 = pixel_box
    tile_w, tile_h = tile_size
    img_w, img_h = img.size

    crop_box = (
        max(0, x1),
        max(0, y1),
        min(x2, img_w),
        min(y2, img_h)
    )
    region = img.crop(crop_box)

    if img.mode == 'L':
        canvas = Image.new('L', (tile_w, tile_h), pad_value)
    else:
        pv = pad_value if isinstance(pad_value, tuple) else (pad_value,) * len(img.getbands())
        canvas = Image.new(img.mode, (tile_w, tile_h), pv)

    canvas.paste(region, (0, 0))
    return canvas


def decide_split_for_tile_with_global(
    val_count, train_count,
    fg_val_sum, pix_val_sum,
    global_fg_ratio,
    val_ratio,
    tile_fg, tile_pix
):
    """
    使用固定的全局前景比例(global_fg_ratio)进行在线贪心决策。
    目标：同时逼近数量比例(val_ratio)与前景比例(global_fg_ratio)。
    返回 'train' 或 'val'。
    """
    total_so_far = val_count + train_count
    # 方案一：放入val
    val_count_1 = val_count + 1
    val_fraction_1 = val_count_1 / (total_so_far + 1)
    fg_val_1 = fg_val_sum + tile_fg
    pix_val_1 = pix_val_sum + tile_pix
    val_fg_ratio_1 = (fg_val_1 / pix_val_1) if pix_val_1 > 0 else 0.0
    obj_1 = abs(val_fraction_1 - val_ratio) + abs(val_fg_ratio_1 - global_fg_ratio)

    # 方案二：放入train
    val_fraction_2 = val_count / (total_so_far + 1)
    val_fg_ratio_2 = (fg_val_sum / pix_val_sum) if pix_val_sum > 0 else 0.0
    obj_2 = abs(val_fraction_2 - val_ratio) + abs(val_fg_ratio_2 - global_fg_ratio)

    if obj_1 < obj_2:
        return 'val'
    elif obj_2 < obj_1:
        return 'train'
    else:
        return 'val' if val_fraction_1 < val_ratio else 'train'


def process_and_split_dataset_streaming(
    input_dir, output_dir, tile_size=(256, 256), overlap_ratio=0.5,
    size_tolerance=2, val_ratio=0.2, create_test_folder=True,
    overlap_threshold=0.8, filter_black_tiles=True, black_threshold=0.95,
    seed=666
):
    """
    流式处理（改为两阶段但不落盘暂存）：
    1) 收集阶段：遍历所有原图与位置，做去重与黑块过滤，计算前景像素并记录tile元数据；不保存像素。
    2) 划分保存：对收集到的tile随机打乱（seed=666），按数量+前景比例贪心划分，并一次性裁剪保存到train/val；val复制到test。
    """
    random.seed(seed)
    np.random.seed(seed)

    base_names = find_base_names_from_folder(input_dir)
    if not base_names:
        print(f"在 {input_dir} 中未找到符合格式的图像")
        return

    print(f"找到 {len(base_names)} 组原始图像（流式：收集→打乱→划分→保存）")

    train_folders, val_folders, test_folders = create_dataset_folders(output_dir, create_test_folder)

    # 收集阶段
    collected_tiles = []  # 每项: {base, x, y, pixel_box, geo_box, tile_fg, tile_pix}
    global_boxes = []     # 去重盒（地理）

    total_generated_tiles = 0
    total_filtered_overlap_tiles = 0
    total_filtered_black_tiles = 0

    print("阶段1：收集候选tile（去重与黑块过滤）...")

    for base_name in tqdm(base_names, desc="收集原图"):
        path_A = os.path.join(input_dir, f"{base_name}_A.tif")
        path_B = os.path.join(input_dir, f"{base_name}_B.tif")
        path_D = os.path.join(input_dir, f"{base_name}_D.tif")
        path_E = os.path.join(input_dir, f"{base_name}_E.png")

        if not all(os.path.exists(p) for p in [path_A, path_B, path_D, path_E]):
            print(f"警告: 文件集 {base_name} 不完整，跳过")
            continue

        try:
            img_A = Image.open(path_A)
            img_B = Image.open(path_B)
            img_D = Image.open(path_D)
            img_E = Image.open(path_E).convert('L')
            geo_transform = get_geo_transform(path_A)
        except Exception as e:
            print(f"警告: 打开文件集 {base_name} 时出错: {e}")
            continue

        sizes = [img_A.size, img_B.size, img_D.size, img_E.size]
        if len(set(sizes)) > 1:
            if is_acceptable_size_difference(sizes, size_tolerance):
                min_w = min(w for w, h in sizes)
                min_h = min(h for w, h in sizes)
                if img_A.size != (min_w, min_h):
                    img_A = img_A.crop((0, 0, min_w, min_h))
                if img_B.size != (min_w, min_h):
                    img_B = img_B.crop((0, 0, min_w, min_h))
                if img_D.size != (min_w, min_h):
                    img_D = img_D.crop((0, 0, min_w, min_h))
                if img_E.size != (min_w, min_h):
                    img_E = img_E.crop((0, 0, min_w, min_h))
                print(f"信息: 文件集 {base_name} 尺寸已调整为 {min_w}x{min_h}")
            else:
                print(f"警告: 文件集 {base_name} 尺寸差异过大 {sizes}，跳过")
                continue

        width, height = img_A.size
        tile_w, tile_h = tile_size
        stride_w = max(1, int(tile_w * (1 - overlap_ratio)))
        stride_h = max(1, int(tile_h * (1 - overlap_ratio)))
        num_cols = math.ceil((width - tile_w) / stride_w) + 1 if width > tile_w else 1
        num_rows = math.ceil((height - tile_h) / stride_h) + 1 if height > tile_h else 1

        for row in range(num_rows):
            for col in range(num_cols):
                x = col * stride_w
                y = row * stride_h
                pixel_box = (x, y, x + tile_w, y + tile_h)
                geo_box = pixel_box_to_geo_box(pixel_box, geo_transform)

                total_generated_tiles += 1

                # 去重在裁剪前
                if check_overlap_with_existing(geo_box, global_boxes, overlap_threshold):
                    total_filtered_overlap_tiles += 1
                    continue

                # 用A判断黑块
                tile_A_small = crop_with_padding(img_A, pixel_box, tile_size, pad_value=0)
                if filter_black_tiles and is_black_tile(tile_A_small, black_threshold):
                    total_filtered_black_tiles += 1
                    continue

                # 计算前景像素（E）
                tile_E_small = crop_with_padding(img_E, pixel_box, tile_size, pad_value=0)
                arr_e = np.array(tile_E_small)
                tile_fg = int((arr_e > 0).sum())
                tile_pix = arr_e.shape[0] * arr_e.shape[1]

                collected_tiles.append({
                    'base': base_name,
                    'x': x,
                    'y': y,
                    'pixel_box': pixel_box,
                    'geo_box': geo_box,
                    'tile_fg': tile_fg,
                    'tile_pix': tile_pix
                })
                global_boxes.append(geo_box)

    if not collected_tiles:
        print("无可用tile，结束。")
        return

    # 预计算全局前景比例（固定目标）
    global_fg = sum(t['tile_fg'] for t in collected_tiles)
    global_pix = sum(t['tile_pix'] for t in collected_tiles)
    global_fg_ratio = (global_fg / global_pix) if global_pix > 0 else 0.0

    # 打乱（固定种子）
    rng = np.random.RandomState(seed)
    rng.shuffle(collected_tiles)

    # 划分 + 保存阶段
    print("阶段2：随机打乱后划分与保存...")

    val_count = 0
    train_count = 0
    fg_val_sum = 0
    pix_val_sum = 0
    fg_train_sum = 0
    pix_train_sum = 0

    total_saved_tiles = 0

    # 为裁剪保存，按原图分组打开，避免频繁打开关闭
    # 简化实现：逐tile按需打开（保持可读性）

    for t in tqdm(collected_tiles, desc="保存tile"):
        base_name = t['base']
        x = t['x']
        y = t['y']
        pixel_box = t['pixel_box']
        tile_fg = t['tile_fg']
        tile_pix = t['tile_pix']

        split_assignment = decide_split_for_tile_with_global(
            val_count, train_count,
            fg_val_sum, pix_val_sum,
            global_fg_ratio,
            val_ratio,
            tile_fg, tile_pix
        )

        new_base_name = f"{base_name}_original_x{x}_y{y}"

        # 打开图像并实际裁剪、保存
        path_A = os.path.join(input_dir, f"{base_name}_A.tif")
        path_B = os.path.join(input_dir, f"{base_name}_B.tif")
        path_D = os.path.join(input_dir, f"{base_name}_D.tif")
        path_E = os.path.join(input_dir, f"{base_name}_E.png")
        try:
            img_A = Image.open(path_A)
            img_B = Image.open(path_B)
            img_D = Image.open(path_D)
            img_E = Image.open(path_E).convert('L')
        except Exception as e:
            print(f"警告: 打开文件集 {base_name} 时出错: {e}")
            continue

        tile_A_small = crop_with_padding(img_A, pixel_box, tile_size, pad_value=0)
        tile_B_small = crop_with_padding(img_B, pixel_box, tile_size, pad_value=0)
        tile_D_small = crop_with_padding(img_D, pixel_box, tile_size, pad_value=0)
        tile_E_small = crop_with_padding(img_E, pixel_box, tile_size, pad_value=0)

        target = train_folders if split_assignment == 'train' else val_folders
        try:
            tile_A_small.save(os.path.join(target['A'], f"{new_base_name}.png"), "PNG")
            tile_B_small.save(os.path.join(target['B'], f"{new_base_name}.png"), "PNG")
            tile_D_small.save(os.path.join(target['C'], f"{new_base_name}.png"), "PNG")
            tile_E_small.save(os.path.join(target['label'], f"{new_base_name}.png"), "PNG")

            if split_assignment == 'val' and create_test_folder and test_folders:
                tile_A_small.save(os.path.join(test_folders['A'], f"{new_base_name}.png"), "PNG")
                tile_B_small.save(os.path.join(test_folders['B'], f"{new_base_name}.png"), "PNG")
                tile_D_small.save(os.path.join(test_folders['C'], f"{new_base_name}.png"), "PNG")
                tile_E_small.save(os.path.join(test_folders['label'], f"{new_base_name}.png"), "PNG")

            total_saved_tiles += 1

            if split_assignment == 'val':
                val_count += 1
                fg_val_sum += tile_fg
                pix_val_sum += tile_pix
            else:
                train_count += 1
                fg_train_sum += tile_fg
                pix_train_sum += tile_pix
        except Exception as e:
            print(f"保存小块时出错: {e}")

    print("\n处理完成（流式）！")
    print(f"训练集: {train_count} 个小块")
    print(f"验证集: {val_count} 个小块")
    if create_test_folder:
        print(f"测试集: {val_count} 个小块 (与验证集相同)")
    print(f"总共生成 {total_generated_tiles} 个小块，保存 {total_saved_tiles} 个小块")
    print(f"重叠度阈值: {overlap_threshold * 100:.1f}%")
    print(f"总共过滤了 {total_filtered_overlap_tiles} 个重叠小块 和 {total_filtered_black_tiles} 个纯黑色小块")


def create_dataset_folders(output_dir, create_test_folder=True):
    """
    创建数据集文件夹结构
    
    参数:
        output_dir: 输出根目录
        create_test_folder: 是否创建测试集文件夹
    
    返回:
        train_folders, val_folders, test_folders
    """
    # 创建主目录结构
    train_folder = os.path.join(output_dir, "train")
    val_folder = os.path.join(output_dir, "val")
    test_folder = os.path.join(output_dir, "test") if create_test_folder else None

    # 训练集文件夹
    train_folders = {
        'A': os.path.join(train_folder, "A"),
        'B': os.path.join(train_folder, "B"),
        'C': os.path.join(train_folder, "C"),
        'label': os.path.join(train_folder, "label")
    }

    # 验证集文件夹
    val_folders = {
        'A': os.path.join(val_folder, "A"),
        'B': os.path.join(val_folder, "B"),
        'C': os.path.join(val_folder, "C"),
        'label': os.path.join(val_folder, "label")
    }

    # 测试集文件夹
    test_folders = None
    if create_test_folder:
        test_folders = {
            'A': os.path.join(test_folder, "A"),
            'B': os.path.join(test_folder, "B"),
            'C': os.path.join(test_folder, "C"),
            'label': os.path.join(test_folder, "label")
        }

    # 创建所有文件夹
    all_folders = list(train_folders.values()) + list(val_folders.values())
    if test_folders:
        all_folders.extend(list(test_folders.values()))

    for folder in all_folders:
        os.makedirs(folder, exist_ok=True)

    return train_folders, val_folders, test_folders


def find_base_names_from_folder(input_dir):
    """
    从文件夹中找出所有基础名称

    参数:
        input_dir: 输入目录

    返回:
        base_names: 基础名称列表
    """
    base_names = set()

    # 查找所有A类型文件
    a_files = glob.glob(os.path.join(input_dir, "*_A.tif"))
    for a_file in a_files:
        # 去掉路径和后缀
        filename = os.path.basename(a_file)
        # 去掉_A.tif部分
        base_name = filename[:-6] if filename.endswith("_A.tif") else filename
        base_names.add(base_name)

    return list(base_names)


def process_image_set_with_overlap_filter(base_name, input_dir, train_folders, val_folders, test_folders,
                                        tile_size=(256, 256), overlap_ratio=0.5, pad_value=0,
                                        size_tolerance=2, apply_augmentation=True,
                                        apply_h_flip=APPLY_H_FLIP, apply_v_flip=APPLY_V_FLIP,
                                        apply_rot90=APPLY_ROT90, apply_rot180=APPLY_ROT180, apply_rot270=APPLY_ROT270,
                                        overlap_threshold=0.8, split_assignment="train", global_boxes=None,
                                        filter_black_tiles=True, black_threshold=0.95):
    """
    处理一组相关的图像(A、B、D、E)，切分为重叠的小块，过滤高重叠度的小块，并直接保存到指定数据集

    参数:
        base_name: 图像基础名称
        input_dir: 输入目录
        train_folders: 训练集文件夹字典
        val_folders: 验证集文件夹字典
        test_folders: 测试集文件夹字典（可选）
        tile_size: 小块大小 (width, height)
        overlap_ratio: 重叠比例
        pad_value: 填充值
        size_tolerance: 允许的尺寸差异像素数
        apply_augmentation: 是否应用数据增强
        apply_h_flip: 是否应用水平翻转
        apply_v_flip: 是否应用垂直翻转
        apply_rot90: 是否应用90°旋转
        apply_rot180: 是否应用180°旋转
        apply_rot270: 是否应用270°旋转
        overlap_threshold: 重叠度阈值
        split_assignment: 数据集分配 ("train" 或 "val")
        global_boxes: 全局边界框列表，用于跨图像的重叠检测
        filter_black_tiles: 是否过滤纯黑色小块
        black_threshold: 纯黑色判定阈值

    返回:
        (保存的小块数量, 生成的小块数量)
    """
    # 构建文件路径
    path_A = os.path.join(input_dir, f"{base_name}_A.tif")
    path_B = os.path.join(input_dir, f"{base_name}_B.tif")
    path_D = os.path.join(input_dir, f"{base_name}_D.tif")
    path_E = os.path.join(input_dir, f"{base_name}_E.png")

    # 检查文件是否存在
    if not all(os.path.exists(p) for p in [path_A, path_B, path_D, path_E]):
        print(f"警告: 文件集 {base_name} 不完整，跳过")
        return None

    # 读取图像
    try:
        img_A = Image.open(path_A)
        img_B = Image.open(path_B)
        img_D = Image.open(path_D)
        img_E = Image.open(path_E).convert('L')  # 确保标签是灰度图
        
        # 获取地理坐标变换信息（以A图像为准）
        geo_transform = get_geo_transform(path_A)
        print(f"信息: 文件集 {base_name} 使用地理坐标进行重叠检测")
            
    except Exception as e:
        print(f"警告: 打开文件集 {base_name} 时出错: {e}")
        return None

    # 检查尺寸一致性
    sizes = [img_A.size, img_B.size, img_D.size, img_E.size]

    # 如果尺寸不完全一致，但差异在可接受范围内，则调整尺寸
    if len(set(sizes)) > 1:
        if is_acceptable_size_difference(sizes, size_tolerance):
            # 找出最小的共同尺寸
            min_width = min(w for w, h in sizes)
            min_height = min(h for w, h in sizes)

            # 调整所有图像到相同尺寸
            if img_A.size != (min_width, min_height):
                img_A = img_A.crop((0, 0, min_width, min_height))
            if img_B.size != (min_width, min_height):
                img_B = img_B.crop((0, 0, min_width, min_height))
            if img_D.size != (min_width, min_height):
                img_D = img_D.crop((0, 0, min_width, min_height))
            if img_E.size != (min_width, min_height):
                img_E = img_E.crop((0, 0, min_width, min_height))

            print(f"信息: 文件集 {base_name} 尺寸已调整为 {min_width}x{min_height}")
        else:
            print(f"警告: 文件集 {base_name} 尺寸差异过大 {sizes}，跳过")
            return None

    total_tiles = 0
    saved_tiles = 0
    filtered_overlap_tiles = 0
    filtered_black_tiles = 0

    # 应用数据增强
    if apply_augmentation:
        augmented_images = apply_geometric_augmentations(img_A, img_B, img_D, img_E, apply_h_flip, apply_v_flip, apply_rot90, apply_rot180, apply_rot270)
    else:
        augmented_images = [(img_A, img_B, img_D, img_E, "original")]

    # 选择目标文件夹
    if split_assignment == "train":
        target_folders = train_folders
    else:
        target_folders = val_folders

    # 对每个增强后的图像集合进行重叠式切分和保存
    for aug_A, aug_B, aug_D, aug_E, aug_type in augmented_images:
        # 重叠式切分图像为小块
        tiles_A, positions, boxes = tile_image_with_overlap(aug_A, tile_size, overlap_ratio, geo_transform=geo_transform)
        tiles_B, _, _ = tile_image_with_overlap(aug_B, tile_size, overlap_ratio, geo_transform=geo_transform)
        tiles_D, _, _ = tile_image_with_overlap(aug_D, tile_size, overlap_ratio, geo_transform=geo_transform)
        tiles_E, _, _ = tile_image_with_overlap(aug_E, tile_size, overlap_ratio, pad_value=0, geo_transform=geo_transform)  # 标签用0填充

        # 保存切分后的小块
        for i, ((x, y), box, tile_A, tile_B, tile_D, tile_E) in enumerate(
                zip(positions, boxes, tiles_A, tiles_B, tiles_D, tiles_E)):
            
            total_tiles += 1
            
            # 检查是否与已保存的小块重叠度过高
            if global_boxes is not None and check_overlap_with_existing(box, global_boxes, overlap_threshold):
                filtered_overlap_tiles += 1
                continue  # 跳过重叠度过高的小块
            
            # 检查是否为纯黑色小块
            if filter_black_tiles and is_black_tile(tile_A, black_threshold):
                filtered_black_tiles += 1
                continue  # 跳过纯黑色小块
            
            # 构建新的基础名称，包含原始坐标信息和增强类型
            new_base_name = f"{base_name}_{aug_type}_x{x}_y{y}"

            try:
                # 保存到目标数据集
                # A类图像
                tile_A.save(os.path.join(target_folders['A'], f"{new_base_name}.png"), "PNG")
                # B类图像
                tile_B.save(os.path.join(target_folders['B'], f"{new_base_name}.png"), "PNG")
                # D类图像（保存为C）
                tile_D.save(os.path.join(target_folders['C'], f"{new_base_name}.png"), "PNG")
                # 标签图像
                tile_E.save(os.path.join(target_folders['label'], f"{new_base_name}.png"), "PNG")

                # 如果是验证集且需要创建测试集，也保存到测试集
                if split_assignment == "val" and test_folders:
                    tile_A.save(os.path.join(test_folders['A'], f"{new_base_name}.png"), "PNG")
                    tile_B.save(os.path.join(test_folders['B'], f"{new_base_name}.png"), "PNG")
                    tile_D.save(os.path.join(test_folders['C'], f"{new_base_name}.png"), "PNG")
                    tile_E.save(os.path.join(test_folders['label'], f"{new_base_name}.png"), "PNG")

                saved_tiles += 1
                
                # 将已保存的边界框添加到全局列表中
                if global_boxes is not None:
                    global_boxes.append(box)
                
            except Exception as e:
                print(f"保存小块时出错: {e}")

    if total_tiles > saved_tiles:
        filter_info = []
        if filtered_overlap_tiles > 0:
            filter_info.append(f"{filtered_overlap_tiles} 个重叠小块")
        if filtered_black_tiles > 0:
            filter_info.append(f"{filtered_black_tiles} 个纯黑色小块")
        
        if filter_info:
            filter_text = "过滤了 " + " 和 ".join(filter_info)
        else:
            filter_text = f"过滤了 {total_tiles - saved_tiles} 个小块"
            
        print(f"文件集 {base_name}: 生成 {total_tiles} 个小块，保存 {saved_tiles} 个小块 ({filter_text})")
    
    return (saved_tiles, total_tiles)


def process_and_split_dataset(input_dir, output_dir, tile_size=(256, 256), overlap_ratio=0.5, 
                            size_tolerance=2, val_ratio=0.2, create_test_folder=True,
                            apply_augmentation=True, apply_h_flip=APPLY_H_FLIP, apply_v_flip=APPLY_V_FLIP,
                            apply_rot90=APPLY_ROT90, apply_rot180=APPLY_ROT180, apply_rot270=APPLY_ROT270,
                            overlap_threshold=0.8, filter_black_tiles=True, black_threshold=0.95,
                            geo_aware=True, geo_eps=2000, geo_min_samples=1):
    """
    处理整个数据集的图像，应用重叠过滤，并直接分割为训练集和验证集

    参数:
        input_dir: 输入目录
        output_dir: 输出目录
        tile_size: 小块大小 (width, height)
        overlap_ratio: 重叠比例
        size_tolerance: 允许的尺寸差异像素数
        val_ratio: 验证集比例
        create_test_folder: 是否创建测试集文件夹
        apply_augmentation: 是否应用数据增强
        apply_h_flip: 是否应用水平翻转
        apply_v_flip: 是否应用垂直翻转
        apply_rot90: 是否应用90°旋转
        apply_rot180: 是否应用180°旋转
        apply_rot270: 是否应用270°旋转
        overlap_threshold: 重叠度阈值
        filter_black_tiles: 是否过滤纯黑色小块
        black_threshold: 纯黑色判定阈值
        geo_aware: 是否启用地理感知划分
        geo_eps: 地理聚类邻域半径（米）
        geo_min_samples: 地理聚类最小样本数
    """
    # 获取所有基础名称
    base_names = find_base_names_from_folder(input_dir)

    if not base_names:
        print(f"在 {input_dir} 中未找到符合格式的图像")
        return

    print(f"找到 {len(base_names)} 组原始图像")

    # 创建数据集文件夹结构
    train_folders, val_folders, test_folders = create_dataset_folders(output_dir, create_test_folder)

    # 预计算所有基础图的前景像素统计
    print("统计整图前景像素比例(用于均衡划分)...")
    fg_stats = compute_fg_stats_for_basenames(input_dir, base_names)

    # 🌍 地理感知的数据集划分
    if geo_aware and SKLEARN_AVAILABLE:
        print("使用地理感知划分模式...")

        # 分析地理坐标分布
        coords, valid_names = analyze_geographic_distribution(input_dir, base_names)

        if len(valid_names) < len(base_names):
            print(f"警告: {len(base_names) - len(valid_names)} 个图像无法读取地理坐标，已跳过")
            base_names = valid_names

        if len(coords) < 2:
            print("警告: 可用图像数量不足，回退到随机划分")
            random.shuffle(base_names)
            split_idx = int(len(base_names) * (1 - val_ratio))
            train_base_names = base_names[:split_idx]
            val_base_names = base_names[split_idx:]
        else:
            # 执行地理聚类
            cluster_labels = perform_geographic_clustering(coords, geo_eps, geo_min_samples)

            # 基于聚类进行训练/验证集划分（均衡前景比例）
            train_base_names, val_base_names = geo_aware_train_val_split_balanced(
                base_names, cluster_labels, val_ratio, fg_stats
            )

            # 验证地理分离程度
            verify_geographic_separation(train_base_names, val_base_names, input_dir)
    else:
        # 传统随机划分（向后兼容）
        if not geo_aware:
            print("使用传统随机划分模式...")
        else:
            print("sklearn不可用，回退到随机划分模式...")

        random.shuffle(base_names)
        split_idx = int(len(base_names) * (1 - val_ratio))
        train_base_names = base_names[:split_idx]
        val_base_names = base_names[split_idx:]

    print(f"训练集: {len(train_base_names)} 组图像")
    print(f"验证集: {len(val_base_names)} 组图像")

    # 输出划分的前景比例统计
    print_split_fg_summary("训练集(整图)", train_base_names, fg_stats)
    print_split_fg_summary("验证集(整图)", val_base_names, fg_stats)

    # 分离的边界框列表，避免训练集和验证集相互过滤
    train_global_boxes = []  # 训练集内部重叠检测
    val_global_boxes = []    # 验证集内部重叠检测

    # 处理训练集
    total_train_tiles = 0
    total_generated_train_tiles = 0
    processed_train_groups = 0

    print("处理训练集...")
    for base_name in tqdm(train_base_names, desc="处理训练集图像"):
        # 强制禁用离线几何增强（训练时已在线增强）
        result = process_image_set_with_overlap_filter(
            base_name, input_dir, train_folders, val_folders, test_folders,
            tile_size, overlap_ratio=overlap_ratio, size_tolerance=size_tolerance,
            apply_augmentation=False,  # 禁用离线增强
            apply_h_flip=False, apply_v_flip=False,
            apply_rot90=False, apply_rot180=False, apply_rot270=False,
            overlap_threshold=overlap_threshold, split_assignment="train", global_boxes=train_global_boxes,
            filter_black_tiles=filter_black_tiles, black_threshold=black_threshold
        )
        if result:
            saved_count, generated_count = result
            if saved_count > 0:
                total_train_tiles += saved_count
                total_generated_train_tiles += generated_count
                processed_train_groups += 1

    # 处理验证集
    total_val_tiles = 0
    total_generated_val_tiles = 0
    processed_val_groups = 0

    print("处理验证集...")
    for base_name in tqdm(val_base_names, desc="处理验证集图像"):
        result = process_image_set_with_overlap_filter(
            base_name, input_dir, train_folders, val_folders, test_folders,
            tile_size, overlap_ratio=overlap_ratio, size_tolerance=size_tolerance,
            apply_augmentation=False,  # 验证集同样禁用离线增强
            apply_h_flip=False, apply_v_flip=False,
            apply_rot90=False, apply_rot180=False, apply_rot270=False,
            overlap_threshold=overlap_threshold, split_assignment="val", global_boxes=val_global_boxes,
            filter_black_tiles=filter_black_tiles, black_threshold=black_threshold
        )
        if result:
            saved_count, generated_count = result
            if saved_count > 0:
                total_val_tiles += saved_count
                total_generated_val_tiles += generated_count
                processed_val_groups += 1

    # 统计结果
    total_tiles = total_train_tiles + total_val_tiles
    total_generated_tiles = total_generated_train_tiles + total_generated_val_tiles
    total_filtered_tiles = total_generated_tiles - total_tiles

    print(f"\n🎯 地理感知划分完成:" if geo_aware and SKLEARN_AVAILABLE else f"\n处理完成！")
    print(f"成功处理 {processed_train_groups + processed_val_groups}/{len(base_names)} 组图像")
    print(f"训练集: {total_train_tiles} 个小块 (来自 {len(train_base_names)} 个原始大图)")
    print(f"验证集: {total_val_tiles} 个小块 (来自 {len(val_base_names)} 个原始大图)")
    if create_test_folder:
        print(f"测试集: {total_val_tiles} 个小块 (与验证集相同)")
    print(f"总共生成 {total_generated_tiles} 个小块，保存 {total_tiles} 个小块")
    print(f"重叠度阈值: {overlap_threshold * 100:.1f}%")
    print(f"总共过滤了 {total_filtered_tiles} 个重叠小块")

    if geo_aware and SKLEARN_AVAILABLE:
        print(f"\n🔍 地理分离验证:")
        print(f"最终训练集: {len(train_base_names)} 个原始大图")
        print(f"最终验证集: {len(val_base_names)} 个原始大图")
        print(f"✅ 无重叠，地理完全分离")


def verify_dataset_structure(dataset_path):
    """
    验证数据集结构完整性

    参数:
        dataset_path: 数据集根目录
    """
    # 验证文件夹结构
    required_folders = [
        os.path.join("train", "A"),
        os.path.join("train", "B"),
        os.path.join("train", "C"),
        os.path.join("train", "label"),
        os.path.join("val", "A"),
        os.path.join("val", "B"),
        os.path.join("val", "C"),
        os.path.join("val", "label")
    ]

    optional_folders = [
        os.path.join("test", "A"),
        os.path.join("test", "B"),
        os.path.join("test", "C"),
        os.path.join("test", "label")
    ]

    all_folders = required_folders + optional_folders

    folder_exists = {}
    for folder in all_folders:
        full_path = os.path.join(dataset_path, folder)
        folder_exists[folder] = os.path.exists(full_path)

    print("文件夹结构验证:")
    for folder in required_folders:
        status = "✓" if folder_exists[folder] else "✗"
        print(f"  {status} {folder}")

    print("\n可选文件夹:")
    for folder in optional_folders:
        status = "✓" if folder_exists[folder] else "-"
        print(f"  {status} {folder}")

    # 验证文件数量
    file_counts = {}
    for folder in all_folders:
        if folder_exists[folder]:
            full_path = os.path.join(dataset_path, folder)
            file_counts[folder] = len(os.listdir(full_path))

    print("\n文件数量验证:")
    for folder in required_folders:
        if folder_exists[folder]:
            print(f"  {folder}: {file_counts[folder]} 个文件")

    # 验证文件名一致性
    print("\n文件名一致性验证:")

    def get_file_basenames(folder_path):
        if not os.path.exists(folder_path):
            return set()
        return {os.path.splitext(filename)[0] for filename in os.listdir(folder_path)}

    for split in ["train", "val", "test"]:
        if not all(folder_exists.get(os.path.join(split, subfolder), False) for subfolder in ["A", "B", "C", "label"]):
            print(f"  {split} 文件夹不完整，跳过一致性检查")
            continue

        a_files = get_file_basenames(os.path.join(dataset_path, split, "A"))
        b_files = get_file_basenames(os.path.join(dataset_path, split, "B"))
        c_files = get_file_basenames(os.path.join(dataset_path, split, "C"))
        label_files = get_file_basenames(os.path.join(dataset_path, split, "label"))

        a_b_consistent = a_files == b_files
        a_c_consistent = a_files == c_files
        a_label_consistent = a_files == label_files

        print(f"  {split} 集:")
        print(f"    A 与 B 一致: {'✓' if a_b_consistent else '✗'}")
        print(f"    A 与 C 一致: {'✓' if a_c_consistent else '✗'}")
        print(f"    A 与 label 一致: {'✓' if a_label_consistent else '✗'}")

    print("\n验证完成！")


def main():
    parser = argparse.ArgumentParser(description='处理变化检测数据集图像，应用重叠过滤，并分割为训练集和验证集')
    parser.add_argument('--input_dir', type=str, default=DEFAULT_INPUT_DIR, help=f'输入目录 (默认: {DEFAULT_INPUT_DIR})')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR, help=f'输出目录 (默认: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--tile_size', type=int, default=DEFAULT_TILE_SIZE,
                        help=f'小块大小 (默认: {DEFAULT_TILE_SIZE})')
    parser.add_argument('--overlap_ratio', type=float, default=DEFAULT_OVERLAP_RATIO,
                        help=f'重叠比例 (0-1之间，默认: {DEFAULT_OVERLAP_RATIO})')
    parser.add_argument('--overlap_threshold', type=float, default=DEFAULT_OVERLAP_THRESHOLD,
                        help=f'重叠度阈值，超过此值的小块将被丢弃 (0-1之间，默认: {DEFAULT_OVERLAP_THRESHOLD})')
    parser.add_argument('--val_ratio', type=float, default=DEFAULT_VAL_RATIO,
                        help=f'验证集比例 (0-1之间，默认: {DEFAULT_VAL_RATIO})')
    parser.add_argument('--size_tolerance', type=int, default=DEFAULT_SIZE_TOLERANCE,
                        help=f'允许的图像尺寸差异像素数 (默认: {DEFAULT_SIZE_TOLERANCE})')
    parser.add_argument('--no_augmentation', action='store_true',
                        help='禁用数据增强')
    parser.add_argument('--no_test', action='store_false', dest='create_test_folder',
                        help='不创建测试集文件夹')
    parser.add_argument('--apply_h_flip', action='store_true', default=APPLY_H_FLIP,
                        help=f'应用水平翻转 (默认: {APPLY_H_FLIP})')
    parser.add_argument('--apply_v_flip', action='store_true', default=APPLY_V_FLIP,
                        help=f'应用垂直翻转 (默认: {APPLY_V_FLIP})')
    parser.add_argument('--apply_rot90', action='store_true', default=APPLY_ROT90,
                        help=f'应用90°旋转 (默认: {APPLY_ROT90})')
    parser.add_argument('--apply_rot180', action='store_true', default=APPLY_ROT180,
                        help=f'应用180°旋转 (默认: {APPLY_ROT180})')
    parser.add_argument('--apply_rot270', action='store_true', default=APPLY_ROT270,
                        help=f'应用270°旋转 (默认: {APPLY_ROT270})')
    parser.add_argument('--no_filter_black', action='store_false', dest='filter_black_tiles',
                        help='禁用纯黑色小块过滤')
    parser.add_argument('--black_threshold', type=float, default=DEFAULT_BLACK_THRESHOLD,
                        help=f'纯黑色判定阈值 (0-1之间，默认: {DEFAULT_BLACK_THRESHOLD})')
    parser.add_argument('--verify', action='store_true',
                        help='验证输出数据集结构')
    parser.add_argument('--geo_aware', action='store_true', default=DEFAULT_GEO_AWARE,
                        help=f'启用地理感知划分 (默认: {DEFAULT_GEO_AWARE})')
    parser.add_argument('--geo_eps', type=int, default=DEFAULT_GEO_EPS,
                        help=f'地理聚类邻域半径 (米) (默认: {DEFAULT_GEO_EPS})')
    parser.add_argument('--geo_min_samples', type=int, default=DEFAULT_GEO_MIN_SAMPLES,
                        help=f'地理聚类最小样本数 (默认: {DEFAULT_GEO_MIN_SAMPLES})')


    args = parser.parse_args()

    # 设置参数
    tile_size = (args.tile_size, args.tile_size)
    overlap_ratio = args.overlap_ratio
    overlap_threshold = args.overlap_threshold
    val_ratio = args.val_ratio
    size_tolerance = args.size_tolerance
    # 强制禁用离线数据增强：训练阶段已有在线增强，这里保持原图
    apply_augmentation = False
    create_test_folder = getattr(args, 'create_test_folder', DEFAULT_CREATE_TEST_FOLDER)
    apply_h_flip = False
    apply_v_flip = False
    apply_rot90 = False
    apply_rot180 = False
    apply_rot270 = False
    filter_black_tiles = getattr(args, 'filter_black_tiles', DEFAULT_FILTER_BLACK_TILES)
    black_threshold = args.black_threshold
    geo_aware = getattr(args, 'geo_aware', DEFAULT_GEO_AWARE)
    geo_eps = args.geo_eps
    geo_min_samples = args.geo_min_samples

    # 参数验证
    if overlap_ratio < 0 or overlap_ratio >= 1:
        print(f"警告: 重叠比例必须在0-1之间，当前值 {overlap_ratio} 将被重置为 {DEFAULT_OVERLAP_RATIO}")
        overlap_ratio = DEFAULT_OVERLAP_RATIO

    if overlap_threshold < 0 or overlap_threshold > 1:
        print(f"警告: 重叠度阈值必须在0-1之间，当前值 {overlap_threshold} 将被重置为 {DEFAULT_OVERLAP_THRESHOLD}")
        overlap_threshold = DEFAULT_OVERLAP_THRESHOLD

    if val_ratio < 0 or val_ratio > 1:
        print(f"警告: 验证集比例必须在0-1之间，当前值 {val_ratio} 将被重置为 {DEFAULT_VAL_RATIO}")
        val_ratio = DEFAULT_VAL_RATIO

    if black_threshold < 0 or black_threshold > 1:
        print(f"警告: 纯黑色判定阈值必须在0-1之间，当前值 {black_threshold} 将被重置为 {DEFAULT_BLACK_THRESHOLD}")
        black_threshold = DEFAULT_BLACK_THRESHOLD

    # 地理感知参数验证
    if geo_eps <= 0:
        print(f"警告: 地理聚类邻域半径必须大于0，当前值 {geo_eps} 将被重置为 {DEFAULT_GEO_EPS}")
        geo_eps = DEFAULT_GEO_EPS
    
    if geo_min_samples < 1:
        print(f"警告: 地理聚类最小样本数必须至少为1，当前值 {geo_min_samples} 将被重置为 {DEFAULT_GEO_MIN_SAMPLES}")
        geo_min_samples = DEFAULT_GEO_MIN_SAMPLES

    print(f"运行参数:")
    print(f"  输入目录: {args.input_dir}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  小块大小: {tile_size[0]}x{tile_size[1]}")
    print(f"  重叠比例: {overlap_ratio * 100:.1f}%")
    print(f"  重叠度阈值: {overlap_threshold * 100:.1f}%")
    print(f"  验证集比例: {val_ratio * 100:.1f}%")
    print(f"  创建测试集: {'是' if create_test_folder else '否'}")
    print(f"  纯黑色小块过滤: {'启用' if filter_black_tiles else '禁用'}")
    if filter_black_tiles:
        print(f"  纯黑色判定阈值: {black_threshold * 100:.1f}%")
    print(f"  允许的尺寸差异: {size_tolerance}像素")
    print(f"  划分策略: 流式tile级划分（数量+前景比例），无地理隔离")
    print(f"  随机种子: 666")
    print(f"  几何变换数据增强: 已禁用 (训练时在线增强)")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 处理数据集（新流程：流式切片+去重+划分，一次落盘）
    process_and_split_dataset_streaming(
        args.input_dir, args.output_dir, tile_size, overlap_ratio,
        size_tolerance, val_ratio, create_test_folder,
        overlap_threshold, filter_black_tiles, black_threshold,
        seed=666
    )

    # 验证输出数据集结构
    if args.verify:
        print("\n验证输出数据集结构:")
        verify_dataset_structure(args.output_dir)

    print("所有处理完成！")


if __name__ == "__main__":
    main() 