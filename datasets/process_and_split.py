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

# 默认参数设置
DEFAULT_INPUT_DIR = r"D:\0Program\Datasets\241120\Compare\Datas\Final"  # 输入目录
DEFAULT_OUTPUT_DIR = r"D:\0Program\Datasets\241120\Compare\Datas\Split10"  # 输出目录
DEFAULT_TILE_SIZE = 512  # 切片大小
DEFAULT_SIZE_TOLERANCE = 2  # 大小容差
DEFAULT_OVERLAP_RATIO = 0.0  # 裁剪重叠比例
DEFAULT_OVERLAP_THRESHOLD = 0.8  # 重叠度阈值，超过此值的小块将被丢弃
DEFAULT_VAL_RATIO = 0.2  # 验证集占总数据的比例
DEFAULT_CREATE_TEST_FOLDER = True  # 是否创建测试集文件夹
DEFAULT_FILTER_BLACK_TILES = True  # 是否过滤纯黑色小块
DEFAULT_BLACK_THRESHOLD = 0.95  # 纯黑色判定阈值，超过此比例的黑色像素将被视为纯黑色小块

# 数据增强方法控制
APPLY_H_FLIP = False    # 是否应用水平翻转
APPLY_V_FLIP = False    # 是否应用垂直翻转
APPLY_ROT90 = False     # 是否应用90°旋转
APPLY_ROT180 = False    # 是否应用180°旋转
APPLY_ROT270 = False    # 是否应用270°旋转


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
                            overlap_threshold=0.8, filter_black_tiles=True, black_threshold=0.95):
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
    """
    # 获取所有基础名称
    base_names = find_base_names_from_folder(input_dir)

    if not base_names:
        print(f"在 {input_dir} 中未找到符合格式的图像")
        return

    print(f"找到 {len(base_names)} 组原始图像")

    # 创建数据集文件夹结构
    train_folders, val_folders, test_folders = create_dataset_folders(output_dir, create_test_folder)

    # 随机分割数据集
    random.shuffle(base_names)
    split_idx = int(len(base_names) * (1 - val_ratio))
    train_base_names = base_names[:split_idx]
    val_base_names = base_names[split_idx:]

    print(f"训练集: {len(train_base_names)} 组图像")
    print(f"验证集: {len(val_base_names)} 组图像")

    # 全局边界框列表，用于跨图像的重叠检测
    global_boxes = []

    # 处理训练集
    total_train_tiles = 0
    total_generated_train_tiles = 0
    processed_train_groups = 0

    print("处理训练集...")
    for base_name in tqdm(train_base_names, desc="处理训练集图像"):
        # 修改函数调用以获取生成和保存的小块数量
        result = process_image_set_with_overlap_filter(
            base_name, input_dir, train_folders, val_folders, test_folders,
            tile_size, overlap_ratio=overlap_ratio, size_tolerance=size_tolerance,
            apply_augmentation=apply_augmentation,
            apply_h_flip=apply_h_flip, apply_v_flip=apply_v_flip,
            apply_rot90=apply_rot90, apply_rot180=apply_rot180, apply_rot270=apply_rot270,
            overlap_threshold=overlap_threshold, split_assignment="train", global_boxes=global_boxes,
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
            apply_augmentation=apply_augmentation,
            apply_h_flip=apply_h_flip, apply_v_flip=apply_v_flip,
            apply_rot90=apply_rot90, apply_rot180=apply_rot180, apply_rot270=apply_rot270,
            overlap_threshold=overlap_threshold, split_assignment="val", global_boxes=global_boxes,
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
    
    print(f"\n处理完成！")
    print(f"成功处理 {processed_train_groups + processed_val_groups}/{len(base_names)} 组图像")
    print(f"训练集: {total_train_tiles} 个小块")
    print(f"验证集: {total_val_tiles} 个小块")
    if create_test_folder:
        print(f"测试集: {total_val_tiles} 个小块 (与验证集相同)")
    print(f"总共生成 {total_generated_tiles} 个小块，保存 {total_tiles} 个小块")
    print(f"重叠度阈值: {overlap_threshold * 100:.1f}%")
    print(f"总共过滤了 {total_filtered_tiles} 个重叠小块")


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

    args = parser.parse_args()

    # 设置参数
    tile_size = (args.tile_size, args.tile_size)
    overlap_ratio = args.overlap_ratio
    overlap_threshold = args.overlap_threshold
    val_ratio = args.val_ratio
    size_tolerance = args.size_tolerance
    apply_augmentation = not args.no_augmentation
    create_test_folder = getattr(args, 'create_test_folder', DEFAULT_CREATE_TEST_FOLDER)
    apply_h_flip = args.apply_h_flip
    apply_v_flip = args.apply_v_flip
    apply_rot90 = args.apply_rot90
    apply_rot180 = args.apply_rot180
    apply_rot270 = args.apply_rot270
    filter_black_tiles = getattr(args, 'filter_black_tiles', DEFAULT_FILTER_BLACK_TILES)
    black_threshold = args.black_threshold

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
    print(f"  几何变换数据增强: {'已启用' if apply_augmentation else '已禁用'}")
    if apply_augmentation:
        augmentations = []
        if apply_h_flip:
            augmentations.append("水平翻转")
        if apply_v_flip:
            augmentations.append("垂直翻转")
        if apply_rot90:
            augmentations.append("90°旋转")
        if apply_rot180:
            augmentations.append("180°旋转")
        if apply_rot270:
            augmentations.append("270°旋转")
        if augmentations:
            print(f"  应用的数据增强方法: {', '.join(augmentations)}")
        else:
            print(f"  应用的数据增强方法: 仅保留原始图像")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 处理数据集
    process_and_split_dataset(
        args.input_dir, args.output_dir, tile_size, overlap_ratio, size_tolerance,
        val_ratio, create_test_folder, apply_augmentation,
        apply_h_flip, apply_v_flip, apply_rot90, apply_rot180, apply_rot270,
        overlap_threshold, filter_black_tiles, black_threshold
    )

    # 验证数据集结构
    if args.verify:
        print("\n验证输出数据集结构:")
        verify_dataset_structure(args.output_dir)

    print("所有处理完成！")


if __name__ == "__main__":
    main() 