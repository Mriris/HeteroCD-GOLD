import argparse
import glob
import math
import os

from PIL import Image, ImageOps
from tqdm import tqdm

# 设置默认参数
DEFAULT_INPUT_DIR = "/data/jingwei/yantingxuan/Datasets/Final"  # 输入目录
DEFAULT_OUTPUT_DIR = "/data/jingwei/yantingxuan/Datasets/CityCN/Enhanced"  # 输出目录
DEFAULT_TILE_SIZE = 512  # 切片大小
DEFAULT_SIZE_TOLERANCE = 2  # 大小容差
DEFAULT_APPLY_AUGMENTATION = True  # 是否应用数据增强
DEFAULT_OVERLAP_RATIO = 0.5  # 裁剪重叠比例

# 数据增强方法控制（使用 True/False）
APPLY_H_FLIP = True    # 是否应用水平翻转
APPLY_V_FLIP = True    # 是否应用垂直翻转
APPLY_ROT90 = True     # 是否应用90°旋转
APPLY_ROT180 = True    # 是否应用180°旋转
APPLY_ROT270 = True    # 是否应用270°旋转

def tile_image_with_overlap(img, tile_size, overlap_ratio=0.5, pad_value=0):
    """
    将图像切分为固定大小的小块，使用指定的重叠比例，不足的地方进行填充

    参数:
        img: PIL图像对象
        tile_size: 小块大小 (width, height)
        overlap_ratio: 重叠比例 (0-1之间)
        pad_value: 填充值

    返回:
        tiles: 切分后的小块列表
        positions: 每个小块在原图中的位置 (x, y)
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

    # 切分图像
    for row in range(num_rows):
        for col in range(num_cols):
            x = col * stride_w
            y = row * stride_h

            # 提取小块
            box = (x, y, x + tile_width, y + tile_height)
            tile = padded_img.crop(box)

            tiles.append(tile)
            positions.append((x, y))

    return tiles, positions

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

def process_image_set(base_name, input_dir, output_dir, tile_size=(256, 256), overlap_ratio=0.5, pad_value=0,
                      size_tolerance=2, apply_augmentation=True, apply_h_flip=APPLY_H_FLIP, apply_v_flip=APPLY_V_FLIP,
                      apply_rot90=APPLY_ROT90, apply_rot180=APPLY_ROT180, apply_rot270=APPLY_ROT270):
    """
    处理一组相关的图像(A、B、D、E)，切分为重叠的小块并应用数据增强

    参数:
        base_name: 图像基础名称
        input_dir: 输入目录
        output_dir: 输出目录
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

    返回:
        成功处理的小块数量
    """
    # 构建文件路径
    path_A = os.path.join(input_dir, f"{base_name}_A.tif")
    path_B = os.path.join(input_dir, f"{base_name}_B.tif")
    path_D = os.path.join(input_dir, f"{base_name}_D.tif")
    path_E = os.path.join(input_dir, f"{base_name}_E.png")

    # 检查文件是否存在
    if not all(os.path.exists(p) for p in [path_A, path_B, path_D, path_E]):
        print(f"警告: 文件集 {base_name} 不完整，跳过")
        return 0

    # 读取图像
    try:
        img_A = Image.open(path_A)
        img_B = Image.open(path_B)
        img_D = Image.open(path_D)
        img_E = Image.open(path_E).convert('L')  # 确保标签是灰度图
    except Exception as e:
        print(f"警告: 打开文件集 {base_name} 时出错: {e}")
        return 0

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
            return 0

    total_tiles = 0

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 应用数据增强
    if apply_augmentation:
        augmented_images = apply_geometric_augmentations(img_A, img_B, img_D, img_E, apply_h_flip, apply_v_flip, apply_rot90, apply_rot180, apply_rot270)
    else:
        augmented_images = [(img_A, img_B, img_D, img_E, "original")]

    # 对每个增强后的图像集合进行重叠式切分和保存
    for aug_A, aug_B, aug_D, aug_E, aug_type in augmented_images:
        # 重叠式切分图像为小块
        tiles_A, positions = tile_image_with_overlap(aug_A, tile_size, overlap_ratio)
        tiles_B, _ = tile_image_with_overlap(aug_B, tile_size, overlap_ratio)
        tiles_D, _ = tile_image_with_overlap(aug_D, tile_size, overlap_ratio)
        tiles_E, _ = tile_image_with_overlap(aug_E, tile_size, overlap_ratio, pad_value=0)  # 标签用0填充

        # 保存切分后的小块
        for i, ((x, y), tile_A, tile_B, tile_D, tile_E) in enumerate(
                zip(positions, tiles_A, tiles_B, tiles_D, tiles_E)):
            # 构建新的基础名称，包含原始坐标信息和增强类型
            new_base_name = f"{base_name}_{aug_type}_x{x}_y{y}"

            # 保存路径
            save_path_A = os.path.join(output_dir, f"{new_base_name}_A.tif")
            save_path_B = os.path.join(output_dir, f"{new_base_name}_B.tif")
            save_path_D = os.path.join(output_dir, f"{new_base_name}_D.tif")
            save_path_E = os.path.join(output_dir, f"{new_base_name}_E.png")

            # 保存小块
            tile_A.save(save_path_A)
            tile_B.save(save_path_B)
            tile_D.save(save_path_D)
            tile_E.save(save_path_E)

            total_tiles += 1

    return total_tiles

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

def process_dataset(input_dir, output_dir, tile_size=(256, 256), overlap_ratio=0.5, size_tolerance=2,
                    apply_augmentation=True, apply_h_flip=APPLY_H_FLIP, apply_v_flip=APPLY_V_FLIP,
                    apply_rot90=APPLY_ROT90, apply_rot180=APPLY_ROT180, apply_rot270=APPLY_ROT270):
    """
    处理整个数据集的图像

    参数:
        input_dir: 输入目录
        output_dir: 输出目录
        tile_size: 小块大小 (width, height)
        overlap_ratio: 重叠比例
        size_tolerance: 允许的尺寸差异像素数
        apply_augmentation: 是否应用数据增强
        apply_h_flip: 是否应用水平翻转
        apply_v_flip: 是否应用垂直翻转
        apply_rot90: 是否应用90°旋转
        apply_rot180: 是否应用180°旋转
        apply_rot270: 是否应用270°旋转
    """
    # 获取所有基础名称
    base_names = find_base_names_from_folder(input_dir)

    if not base_names:
        print(f"在 {input_dir} 中未找到符合格式的图像")
        return

    print(f"找到 {len(base_names)} 组原始图像")

    # 处理每组图像
    total_tiles = 0
    processed_groups = 0

    for base_name in tqdm(base_names, desc="处理图像"):
        tiles_count = process_image_set(base_name, input_dir, output_dir, tile_size,
                                        overlap_ratio=overlap_ratio,
                                        size_tolerance=size_tolerance,
                                        apply_augmentation=apply_augmentation,
                                        apply_h_flip=apply_h_flip,
                                        apply_v_flip=apply_v_flip,
                                        apply_rot90=apply_rot90,
                                        apply_rot180=apply_rot180,
                                        apply_rot270=apply_rot270)
        if tiles_count > 0:
            total_tiles += tiles_count
            processed_groups += 1

    print(f"成功处理 {processed_groups}/{len(base_names)} 组图像")
    print(f"总共生成 {total_tiles} 个小块")

def main():
    parser = argparse.ArgumentParser(description='将变化检测数据集的图像裁剪为小块并应用几何变换数据增强')
    parser.add_argument('--input_dir', type=str, help=f'输入目录 (默认: {DEFAULT_INPUT_DIR})')
    parser.add_argument('--output_dir', type=str, help=f'输出目录 (默认: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--tile_size', type=int, default=DEFAULT_TILE_SIZE,
                        help=f'小块大小 (默认: {DEFAULT_TILE_SIZE})')
    parser.add_argument('--overlap_ratio', type=float, default=DEFAULT_OVERLAP_RATIO,
                        help=f'重叠比例 (0-1之间，默认: {DEFAULT_OVERLAP_RATIO})')
    parser.add_argument('--size_tolerance', type=int, default=DEFAULT_SIZE_TOLERANCE,
                        help=f'允许的图像尺寸差异像素数 (默认: {DEFAULT_SIZE_TOLERANCE})')
    parser.add_argument('--no_augmentation', action='store_true',
                        help='禁用数据增强')
    parser.add_argument('--apply_h_flip', type=bool, default=APPLY_H_FLIP,
                        help=f'是否应用水平翻转 (默认: {APPLY_H_FLIP})')
    parser.add_argument('--apply_v_flip', type=bool, default=APPLY_V_FLIP,
                        help=f'是否应用垂直翻转 (默认: {APPLY_V_FLIP})')
    parser.add_argument('--apply_rot90', type=bool, default=APPLY_ROT90,
                        help=f'是否应用90°旋转 (默认: {APPLY_ROT90})')
    parser.add_argument('--apply_rot180', type=bool, default=APPLY_ROT180,
                        help=f'是否应用180°旋转 (默认: {APPLY_ROT180})')
    parser.add_argument('--apply_rot270', type=bool, default=APPLY_ROT270,
                        help=f'是否应用270°旋转 (默认: {APPLY_ROT270})')

    args = parser.parse_args()

    # 优先使用命令行参数，如果没有则使用默认值
    input_dir = args.input_dir if args.input_dir is not None else DEFAULT_INPUT_DIR
    output_dir = args.output_dir if args.output_dir is not None else DEFAULT_OUTPUT_DIR
    tile_size = (args.tile_size, args.tile_size)
    overlap_ratio = args.overlap_ratio
    size_tolerance = args.size_tolerance
    apply_augmentation = not args.no_augmentation
    apply_h_flip = args.apply_h_flip
    apply_v_flip = args.apply_v_flip
    apply_rot90 = args.apply_rot90
    apply_rot180 = args.apply_rot180
    apply_rot270 = args.apply_rot270

    # 确保重叠比例在合理范围内
    if overlap_ratio < 0 or overlap_ratio >= 1:
        print(f"警告: 重叠比例必须在0-1之间，当前值 {overlap_ratio} 将被重置为 {DEFAULT_OVERLAP_RATIO}")
        overlap_ratio = DEFAULT_OVERLAP_RATIO

    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"小块大小: {tile_size[0]}x{tile_size[1]}")
    print(f"重叠比例: {overlap_ratio * 100:.1f}%")
    print(f"允许的尺寸差异: {size_tolerance}像素")
    print(f"几何变换数据增强: {'已启用' if apply_augmentation else '已禁用'}")
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
        print(f"应用的数据增强方法: {', '.join(augmentations)}")

    # 处理数据集
    process_dataset(input_dir, output_dir, tile_size, overlap_ratio, size_tolerance, apply_augmentation,
                    apply_h_flip, apply_v_flip, apply_rot90, apply_rot180, apply_rot270)

    print("处理完成！")

if __name__ == "__main__":
    main()
