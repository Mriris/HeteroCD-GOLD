import argparse
import glob
import math
import os
import random
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import rasterio

# sklearn可选，用于DBSCAN聚类
try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("警告: sklearn未安装，地理感知功能将被禁用（将回退为按整图随机划分）")


# ========================= 默认参数 ========================= #
DEFAULT_INPUT_DIR = r"/data/jingwei/yantingxuan/Datasets/CityCN/Final"
DEFAULT_OUTPUT_DIR = r"/data/jingwei/yantingxuan/Datasets/CityCN/Split21"
DEFAULT_TILE_SIZE = 512 # 小块大小
DEFAULT_OVERLAP_RATIO = 0.5 # 重叠比例
DEFAULT_VAL_RATIO = 0.2 # 验证集比例
DEFAULT_SIZE_TOLERANCE = 2 # 允许尺寸差异
DEFAULT_CREATE_TEST_FOLDER = True # 是否创建测试集文件夹
DEFAULT_FILTER_BLACK_TILES = True # 是否过滤纯黑小块
DEFAULT_BLACK_THRESHOLD = 0.95 # 纯黑小块阈值

DEFAULT_GEO_AWARE = True # 是否启用地理感知
DEFAULT_GEO_EPS = 2000 # 地理感知邻域半径
DEFAULT_GEO_MIN_SAMPLES = 1 # 地理感知最小样本数

# 保存目录键名（A/B/C/label），其中C对应输入的D图像
SPLIT_DIR_KEYS = ['A', 'B', 'C', 'label']


# ========================= 基础工具函数 ========================= #

def find_base_names_from_folder(input_dir: str) -> List[str]:
    """
    从输入目录中查找所有基础名称（依据 *_A.tif）
    """
    base_names = set()
    a_files = glob.glob(os.path.join(input_dir, "*_A.tif"))
    for a_file in a_files:
        filename = os.path.basename(a_file)
        base_name = filename[:-6] if filename.endswith("_A.tif") else filename
        base_names.add(base_name)
    return list(base_names)


def get_image_center_coordinates(tif_path: str) -> Tuple[float, float]:
    """
    获取TIF文件的中心地理坐标 (center_x, center_y)。失败时返回 (None, None)。
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


def is_black_tile(tile: Image.Image, threshold: float = 0.95) -> bool:
    """
    判断一个图像小块是否为“近乎纯黑”。
    规则：灰度<=5或RGB三通道<=5的像素比例 >= threshold。
    """
    arr = np.array(tile)
    if arr.ndim == 2:
        total = arr.size
        black = np.sum(arr <= 5)
    else:
        total = arr.shape[0] * arr.shape[1]
        black = np.sum(np.all(arr <= 5, axis=2))
    ratio = black / total if total > 0 else 0.0
    return ratio >= threshold


def crop_with_padding(img: Image.Image, pixel_box: Tuple[int, int, int, int], tile_size: Tuple[int, int], pad_value: int = 0) -> Image.Image:
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


def compute_grid_positions(width: int, height: int, tile_size: Tuple[int, int], overlap_ratio: float) -> List[Tuple[int, int]]:
    """
    计算切片的左上角坐标网格（支持重叠）。
    """
    tile_w, tile_h = tile_size
    stride_w = max(1, int(tile_w * (1 - overlap_ratio)))
    stride_h = max(1, int(tile_h * (1 - overlap_ratio)))

    num_cols = math.ceil((width - tile_w) / stride_w) + 1 if width > tile_w else 1
    num_rows = math.ceil((height - tile_h) / stride_h) + 1 if height > tile_h else 1

    positions = []
    for row in range(num_rows):
        for col in range(num_cols):
            x = col * stride_w
            y = row * stride_h
            positions.append((x, y))
    return positions


# ========================= 两阶段：收集 → 划分 → 保存 ========================= #

def collect_tiles_metadata(
    input_dir: str,
    base_names: List[str],
    tile_size: Tuple[int, int],
    overlap_ratio: float,
    size_tolerance: int,
    filter_black_tiles: bool,
    black_threshold: float
) -> Tuple[Dict[str, dict], Dict[str, Tuple[float, float]]]:
    """
    第一阶段：
      - 遍历所有原始大图，按给定tile大小与重叠生成候选tile
      - 过滤纯黑小块
      - 收集每幅图的tile元数据与聚合统计（tile数、前景像素与总像素）
      - 记录每幅图的地理中心坐标
    注意：此阶段不保存像素，仅保存元数据，便于后续按整图划分并落盘。
    返回：
      images_meta: base_name -> {
          'width': int, 'height': int,
          'tile_boxes': List[pixel_box],
          'tile_count': int, 'fg_sum': int, 'pix_sum': int
      }
      centers: base_name -> (center_x, center_y)
    """
    images_meta: Dict[str, dict] = {}
    centers: Dict[str, Tuple[float, float]] = {}

    total_tiles = 0
    total_kept_tiles = 0

    print("阶段1：遍历切片并筛选（黑块过滤），仅收集元数据，不落盘...")

    for base_name in tqdm(base_names, desc="收集元数据"):
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
        except Exception as e:
            print(f"警告: 打开文件集 {base_name} 时出错: {e}")
            continue

        # 尺寸对齐到最小公共尺寸
        sizes = [img_A.size, img_B.size, img_D.size, img_E.size]
        if len(set(sizes)) > 1:
            max_width_diff = max(w for w, h in sizes) - min(w for w, h in sizes)
            max_height_diff = max(h for w, h in sizes) - min(h for w, h in sizes)
            if max_width_diff <= size_tolerance and max_height_diff <= size_tolerance:
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
                width, height = min_w, min_h
            else:
                print(f"警告: 文件集 {base_name} 尺寸差异过大 {sizes}，跳过")
                continue
        else:
            width, height = img_A.size

        # 生成候选位置
        positions = compute_grid_positions(width, height, tile_size, overlap_ratio)

        tile_boxes: List[Tuple[int, int, int, int]] = []
        fg_sum = 0
        pix_sum = 0
        kept = 0

        for (x, y) in positions:
            pixel_box = (x, y, x + tile_size[0], y + tile_size[1])
            total_tiles += 1

            # 黑块过滤基于A图
            tile_A = crop_with_padding(img_A, pixel_box, tile_size, pad_value=0)
            if filter_black_tiles and is_black_tile(tile_A, black_threshold):
                continue

            # 统计前景（E图 > 0）
            tile_E = crop_with_padding(img_E, pixel_box, tile_size, pad_value=0)
            arr_e = np.array(tile_E)
            tile_fg = int((arr_e > 0).sum())
            tile_pix = arr_e.shape[0] * arr_e.shape[1]

            fg_sum += tile_fg
            pix_sum += tile_pix
            tile_boxes.append(pixel_box)
            kept += 1

        if kept == 0:
            # 该图像无有效tile
            continue

        total_kept_tiles += kept

        # 地理中心
        cx, cy = get_image_center_coordinates(path_A)
        if cx is None or cy is None:
            # 若无法读取地理坐标，先记录为None，后续聚类时会跳过该图
            pass

        images_meta[base_name] = {
            'width': width,
            'height': height,
            'tile_boxes': tile_boxes,
            'tile_count': kept,
            'fg_sum': fg_sum,
            'pix_sum': pix_sum,
        }
        centers[base_name] = (cx, cy)

    if len(images_meta) == 0:
        print("错误: 无任何有效图像的tile元数据，终止。")
        return images_meta, centers

    # 汇总信息
    global_tiles = sum(v['tile_count'] for v in images_meta.values())
    global_fg = sum(v['fg_sum'] for v in images_meta.values())
    global_pix = sum(v['pix_sum'] for v in images_meta.values())
    global_ratio = (global_fg / global_pix) if global_pix > 0 else 0.0

    print("收集完成：")
    print(f"  候选tile总数: {total_tiles}")
    print(f"  保留tile总数: {total_kept_tiles}")
    print(f"  全局前景比例(像素): {global_fg}/{global_pix} = {global_ratio*100:.2f}%")

    return images_meta, centers


def perform_geographic_clustering_from_centers(
    base_names: List[str],
    centers: Dict[str, Tuple[float, float]],
    eps: int,
    min_samples: int
) -> Dict[str, int]:
    """
    基于每幅图的地理中心坐标进行DBSCAN聚类。
    返回: base_name -> cluster_label
    若sklearn不可用或坐标无效，返回空dict表示无法聚类。
    """
    valid_names = []
    coords = []
    for bn in base_names:
        cx, cy = centers.get(bn, (None, None))
        if cx is not None and cy is not None:
            valid_names.append(bn)
            coords.append([cx, cy])
    if len(valid_names) < 2 or not SKLEARN_AVAILABLE:
        return {}

    coords = np.array(coords)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = clustering.labels_

    mapping = {}
    for name, label in zip(valid_names, labels):
        mapping[name] = int(label)
    return mapping


def split_by_clusters_balanced(
    images_meta: Dict[str, dict],
    cluster_labels: Dict[str, int],
    val_ratio: float
) -> Tuple[List[str], List[str]]:
    """
    按簇（或整图）进行训练/验证划分，目标：
      1) 验证集tile数量接近 val_ratio
      2) 验证集前景比例接近全局比例
    输入统计基于“收集阶段”的tile级统计（fg_sum/pix_sum/tile_count）。
    若 cluster_labels 为空，则退化为整图随机划分（以tile数量为权重贪心）。
    返回：train_base_names, val_base_names
    """
    # 全局目标
    total_tiles = sum(v['tile_count'] for v in images_meta.values())
    target_val_size = int(total_tiles * val_ratio)
    global_fg = sum(v['fg_sum'] for v in images_meta.values())
    global_pix = sum(v['pix_sum'] for v in images_meta.values())
    global_ratio = (global_fg / global_pix) if global_pix > 0 else 0.0

    # 构造簇 -> 成员图像列表
    cluster_to_members: Dict[int, List[str]] = {}
    if len(cluster_labels) > 0:
        for bn, label in cluster_labels.items():
            cluster_to_members.setdefault(label, []).append(bn)
        # 未聚类到的图像（无坐标）作为独立簇
        for bn in images_meta.keys():
            if bn not in cluster_labels:
                cluster_to_members.setdefault(10**9 + hash(bn) % 10**6, []).append(bn)
    else:
        # 无聚类信息：每图单独视为一簇
        for bn in images_meta.keys():
            cluster_to_members[10**9 + hash(bn) % 10**6] = [bn]

    # 聚合每个簇的统计
    cluster_stats = []  # [(label, count, fg, pix, members)]
    for label, members in cluster_to_members.items():
        c_count = sum(images_meta[m]['tile_count'] for m in members)
        c_fg = sum(images_meta[m]['fg_sum'] for m in members)
        c_pix = sum(images_meta[m]['pix_sum'] for m in members)
        cluster_stats.append((label, c_count, c_fg, c_pix, members))

    # 按tile数降序遍历，贪心决定加入验证或训练
    cluster_stats.sort(key=lambda x: x[1], reverse=True)

    train_names: List[str] = []
    val_names: List[str] = []
    val_count = 0
    val_fg = 0
    val_pix = 0

    for label, c_count, c_fg, c_pix, members in cluster_stats:
        if val_count < target_val_size:
            new_val_count = val_count + c_count
            new_val_fg = val_fg + c_fg
            new_val_pix = val_pix + c_pix
            new_val_ratio = (new_val_fg / new_val_pix) if new_val_pix > 0 else 0.0
            curr_val_ratio = (val_fg / val_pix) if val_pix > 0 else 0.0

            improve = abs(new_val_ratio - global_ratio) < abs(curr_val_ratio - global_ratio)
            within = new_val_count <= int(target_val_size * 1.2)

            if within and (improve or (target_val_size - val_count) >= c_count):
                val_names.extend(members)
                val_count, val_fg, val_pix = new_val_count, new_val_fg, new_val_pix
                print(f"  簇 {label}: {c_count} 个tile -> 验证集 (val_fg_ratio={new_val_ratio*100:.2f}%)")
            else:
                train_names.extend(members)
                print(f"  簇 {label}: {c_count} 个tile -> 训练集")
        else:
            train_names.extend(members)
            print(f"  簇 {label}: {c_count} 个tile -> 训练集")

    # 如验证不足，回填小簇
    if val_count < target_val_size:
        need = target_val_size - val_count
        leftovers = [(label, c_count, members) for (label, c_count, _, _, members) in cluster_stats if any(m in train_names for m in members)]
        leftovers.sort(key=lambda x: x[1])
        for label, c_count, members in leftovers:
            if val_count + c_count <= int(target_val_size * 1.2):
                for m in members:
                    if m in train_names:
                        train_names.remove(m)
                val_names.extend(members)
                val_count += c_count
                print(f"  回填簇 {label}: {c_count} 个tile -> 验证集")
            if val_count >= target_val_size:
                break

    # 去重保护
    train_names = sorted(list(set(train_names)))
    val_names = sorted(list(set(val_names)))

    # 防止交叉（理论上不会发生）
    inter = set(train_names) & set(val_names)
    if inter:
        print(f"警告: 划分出现交集 {len(inter)}，将其移至训练集")
        val_names = [n for n in val_names if n not in inter]

    print(f"目标验证集tile数: {target_val_size}/{total_tiles} ({val_ratio*100:.1f}%)")
    print(f"最终 训练: {len(train_names)} 图, 验证: {len(val_names)} 图")
    return train_names, val_names


def create_dataset_folders(output_dir: str, create_test_folder: bool = True):
    """
    创建输出数据集的目录结构。
    返回: train_folders, val_folders, test_folders（字典: A/B/C/label）
    """
    train_folder = os.path.join(output_dir, "train")
    val_folder = os.path.join(output_dir, "val")
    test_folder = os.path.join(output_dir, "test") if create_test_folder else None

    train_folders = {
        'A': os.path.join(train_folder, 'A'),
        'B': os.path.join(train_folder, 'B'),
        'C': os.path.join(train_folder, 'C'),
        'label': os.path.join(train_folder, 'label'),
    }
    val_folders = {
        'A': os.path.join(val_folder, 'A'),
        'B': os.path.join(val_folder, 'B'),
        'C': os.path.join(val_folder, 'C'),
        'label': os.path.join(val_folder, 'label'),
    }

    test_folders = None
    if create_test_folder:
        test_folders = {
            'A': os.path.join(test_folder, 'A'),
            'B': os.path.join(test_folder, 'B'),
            'C': os.path.join(test_folder, 'C'),
            'label': os.path.join(test_folder, 'label'),
        }

    all_dirs = list(train_folders.values()) + list(val_folders.values())
    if test_folders:
        all_dirs += list(test_folders.values())
    for d in all_dirs:
        os.makedirs(d, exist_ok=True)

    return train_folders, val_folders, test_folders


def save_tiles_for_split(
    input_dir: str,
    output_dir: str,
    images_meta: Dict[str, dict],
    train_base_names: List[str],
    val_base_names: List[str],
    tile_size: Tuple[int, int],
    create_test_folder: bool
):
    """
    根据整图划分结果，重新打开原图并依据收集阶段的tile_boxes裁剪并保存。
    说明：
      - A/B/D/E 四类输入，其中D输出目录命名为C以兼容现有训练代码。
      - 验证集复制到测试集。
    返回：统计信息 (train_tiles, val_tiles)
    """
    train_folders, val_folders, test_folders = create_dataset_folders(output_dir, create_test_folder)

    def _save_for_one_split(base_names: List[str], target_folders: dict, also_copy_to_test: bool) -> int:
        saved = 0
        for base_name in tqdm(base_names, desc=("保存验证集" if also_copy_to_test else "保存训练集")):
            meta = images_meta.get(base_name)
            if not meta:
                continue
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

            # 对齐至收集阶段的尺寸（最小公共尺寸）
            width, height = meta['width'], meta['height']
            if img_A.size != (width, height):
                img_A = img_A.crop((0, 0, width, height))
            if img_B.size != (width, height):
                img_B = img_B.crop((0, 0, width, height))
            if img_D.size != (width, height):
                img_D = img_D.crop((0, 0, width, height))
            if img_E.size != (width, height):
                img_E = img_E.crop((0, 0, width, height))

            for pixel_box in meta['tile_boxes']:
                x1, y1, x2, y2 = pixel_box
                base_out_name = f"{base_name}_original_x{x1}_y{y1}"

                tile_A = crop_with_padding(img_A, pixel_box, tile_size, pad_value=0)
                tile_B = crop_with_padding(img_B, pixel_box, tile_size, pad_value=0)
                tile_D = crop_with_padding(img_D, pixel_box, tile_size, pad_value=0)
                tile_E = crop_with_padding(img_E, pixel_box, tile_size, pad_value=0)

                try:
                    tile_A.save(os.path.join(target_folders['A'], f"{base_out_name}.png"), "PNG")
                    tile_B.save(os.path.join(target_folders['B'], f"{base_out_name}.png"), "PNG")
                    tile_D.save(os.path.join(target_folders['C'], f"{base_out_name}.png"), "PNG")
                    tile_E.save(os.path.join(target_folders['label'], f"{base_out_name}.png"), "PNG")

                    if also_copy_to_test and test_folders:
                        tile_A.save(os.path.join(test_folders['A'], f"{base_out_name}.png"), "PNG")
                        tile_B.save(os.path.join(test_folders['B'], f"{base_out_name}.png"), "PNG")
                        tile_D.save(os.path.join(test_folders['C'], f"{base_out_name}.png"), "PNG")
                        tile_E.save(os.path.join(test_folders['label'], f"{base_out_name}.png"), "PNG")

                    saved += 1
                except Exception as e:
                    print(f"保存小块时出错: {e}")
        return saved

    train_tiles = _save_for_one_split(train_base_names, train_folders, also_copy_to_test=False)
    val_tiles = _save_for_one_split(val_base_names, val_folders, also_copy_to_test=True)

    return train_tiles, val_tiles


def verify_dataset_structure(dataset_path: str):
    """
    验证输出数据集结构与文件名一致性。
    """
    required_folders = [
        os.path.join("train", "A"),
        os.path.join("train", "B"),
        os.path.join("train", "C"),
        os.path.join("train", "label"),
        os.path.join("val", "A"),
        os.path.join("val", "B"),
        os.path.join("val", "C"),
        os.path.join("val", "label"),
    ]
    optional_folders = [
        os.path.join("test", "A"),
        os.path.join("test", "B"),
        os.path.join("test", "C"),
        os.path.join("test", "label"),
    ]

    all_folders = required_folders + optional_folders
    folder_exists = {}
    for folder in all_folders:
        full = os.path.join(dataset_path, folder)
        folder_exists[folder] = os.path.exists(full)

    print("文件夹结构验证:")
    for folder in required_folders:
        status = "✓" if folder_exists.get(folder, False) else "✗"
        print(f"  {status} {folder}")

    print("\n可选文件夹:")
    for folder in optional_folders:
        status = "✓" if folder_exists.get(folder, False) else "-"
        print(f"  {status} {folder}")

    file_counts = {}
    for folder in all_folders:
        if folder_exists.get(folder, False):
            full = os.path.join(dataset_path, folder)
            file_counts[folder] = len(os.listdir(full))

    print("\n文件数量验证:")
    for folder in required_folders:
        if folder_exists.get(folder, False):
            print(f"  {folder}: {file_counts[folder]} 个文件")

    print("\n文件名一致性验证:")

    def _basenames(folder_path: str):
        if not os.path.exists(folder_path):
            return set()
        return {os.path.splitext(fn)[0] for fn in os.listdir(folder_path)}

    for split in ["train", "val", "test"]:
        if not all(folder_exists.get(os.path.join(split, sub), False) for sub in ["A", "B", "C", "label"]):
            print(f"  {split} 文件夹不完整，跳过一致性检查")
            continue
        a = _basenames(os.path.join(dataset_path, split, "A"))
        b = _basenames(os.path.join(dataset_path, split, "B"))
        c = _basenames(os.path.join(dataset_path, split, "C"))
        l = _basenames(os.path.join(dataset_path, split, "label"))
        print(f"  {split} 集:")
        print(f"    A 与 B 一致: {'✓' if a == b else '✗'}")
        print(f"    A 与 C 一致: {'✓' if a == c else '✗'}")
        print(f"    A 与 label 一致: {'✓' if a == l else '✗'}")

    print("\n验证完成！")


# ========================= CLI 主流程 ========================= #

def main():
    parser = argparse.ArgumentParser(description='两阶段（先收集后划分）地理优先的变化检测数据预处理：整图/地理簇先划分，再按收集的tile列表落盘')
    parser.add_argument('--input_dir', type=str, default=DEFAULT_INPUT_DIR, help=f'输入目录 (默认: {DEFAULT_INPUT_DIR})')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR, help=f'输出目录 (默认: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--tile_size', type=int, default=DEFAULT_TILE_SIZE, help=f'小块边长 (默认: {DEFAULT_TILE_SIZE})')
    parser.add_argument('--overlap_ratio', type=float, default=DEFAULT_OVERLAP_RATIO, help=f'切片重叠比例 (0-1，默认: {DEFAULT_OVERLAP_RATIO})')
    parser.add_argument('--val_ratio', type=float, default=DEFAULT_VAL_RATIO, help=f'验证集比例 (0-1，默认: {DEFAULT_VAL_RATIO})')
    parser.add_argument('--size_tolerance', type=int, default=DEFAULT_SIZE_TOLERANCE, help=f'允许的尺寸差异像素数 (默认: {DEFAULT_SIZE_TOLERANCE})')
    parser.add_argument('--no_test', action='store_false', dest='create_test_folder', help='不创建测试集文件夹 (默认创建并复制验证集)')
    parser.add_argument('--no_filter_black', action='store_false', dest='filter_black_tiles', help='禁用纯黑小块过滤')
    parser.add_argument('--black_threshold', type=float, default=DEFAULT_BLACK_THRESHOLD, help=f'纯黑判定阈值 (0-1，默认: {DEFAULT_BLACK_THRESHOLD})')
    parser.add_argument('--geo_aware', action='store_true', default=DEFAULT_GEO_AWARE, help=f'启用地理感知簇划分 (默认: {DEFAULT_GEO_AWARE})')
    parser.add_argument('--geo_eps', type=int, default=DEFAULT_GEO_EPS, help=f'DBSCAN 邻域半径(米) (默认: {DEFAULT_GEO_EPS})')
    parser.add_argument('--geo_min_samples', type=int, default=DEFAULT_GEO_MIN_SAMPLES, help=f'DBSCAN 最小样本数 (默认: {DEFAULT_GEO_MIN_SAMPLES})')
    parser.add_argument('--verify', action='store_true', help='落盘后验证输出数据结构')

    args = parser.parse_args()

    tile_size = (args.tile_size, args.tile_size)
    overlap_ratio = args.overlap_ratio
    val_ratio = args.val_ratio
    size_tolerance = args.size_tolerance
    create_test_folder = getattr(args, 'create_test_folder', DEFAULT_CREATE_TEST_FOLDER)
    filter_black_tiles = getattr(args, 'filter_black_tiles', DEFAULT_FILTER_BLACK_TILES)
    black_threshold = args.black_threshold
    geo_aware = getattr(args, 'geo_aware', DEFAULT_GEO_AWARE)
    geo_eps = args.geo_eps
    geo_min_samples = args.geo_min_samples

    # 参数校验与修正
    if not (0 <= overlap_ratio < 1):
        print(f"警告: 重叠比例必须在[0,1)，当前 {overlap_ratio} 已重置为 {DEFAULT_OVERLAP_RATIO}")
        overlap_ratio = DEFAULT_OVERLAP_RATIO
    if not (0 <= val_ratio <= 1):
        print(f"警告: 验证集比例必须在[0,1]，当前 {val_ratio} 已重置为 {DEFAULT_VAL_RATIO}")
        val_ratio = DEFAULT_VAL_RATIO
    if not (0 <= black_threshold <= 1):
        print(f"警告: 纯黑阈值必须在[0,1]，当前 {black_threshold} 已重置为 {DEFAULT_BLACK_THRESHOLD}")
        black_threshold = DEFAULT_BLACK_THRESHOLD
    if geo_eps <= 0:
        print(f"警告: geo_eps 必须>0，当前 {geo_eps} 已重置为 {DEFAULT_GEO_EPS}")
        geo_eps = DEFAULT_GEO_EPS
    if geo_min_samples < 1:
        print(f"警告: geo_min_samples 必须>=1，当前 {geo_min_samples} 已重置为 {DEFAULT_GEO_MIN_SAMPLES}")
        geo_min_samples = DEFAULT_GEO_MIN_SAMPLES

    print("运行参数:")
    print(f"  输入目录: {args.input_dir}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  小块大小: {tile_size[0]}x{tile_size[1]}")
    print(f"  重叠比例: {overlap_ratio*100:.1f}%")
    print(f"  验证集比例: {val_ratio*100:.1f}%")
    print(f"  允许尺寸差异: {size_tolerance} 像素")
    print(f"  纯黑小块过滤: {'启用' if filter_black_tiles else '禁用'} (阈值 {black_threshold*100:.1f}%)")
    print(f"  地理感知簇划分: {'启用' if (geo_aware and SKLEARN_AVAILABLE) else '禁用/不可用'}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 获取基础名称
    base_names = find_base_names_from_folder(args.input_dir)
    if not base_names:
        print(f"在 {args.input_dir} 中未找到符合格式的图像")
        return

    # 阶段1：收集tile元数据
    images_meta, centers = collect_tiles_metadata(
        args.input_dir, base_names, tile_size, overlap_ratio, size_tolerance,
        filter_black_tiles, black_threshold
    )
    if len(images_meta) == 0:
        return

    # 根据是否启用地理感知，构建整图/簇划分
    if geo_aware and SKLEARN_AVAILABLE:
        print("阶段2：基于TIF中心坐标进行DBSCAN聚类并划分...")
        cluster_labels = perform_geographic_clustering_from_centers(
            list(images_meta.keys()), centers, eps=geo_eps, min_samples=geo_min_samples
        )
        if len(cluster_labels) == 0:
            print("警告: 无法进行有效地理聚类，将退化为整图随机划分")
        train_names, val_names = split_by_clusters_balanced(images_meta, cluster_labels, val_ratio)
    else:
        print("阶段2：地理感知不可用/未启用，按整图统计进行随机划分...")
        # 按tile数量作为权重，简单贪心凑够验证集目标
        items = [(bn, v['tile_count']) for bn, v in images_meta.items()]
        items.sort(key=lambda x: x[1], reverse=True)
        total_tiles = sum(c for _, c in items)
        target_val = int(total_tiles * val_ratio)
        val_names, train_names = [], []
        val_count = 0
        for bn, cnt in items:
            if val_count < target_val and (val_count + cnt) <= int(target_val * 1.2):
                val_names.append(bn)
                val_count += cnt
            else:
                train_names.append(bn)
        # 回填不足
        if val_count < target_val:
            for bn, cnt in reversed(items):
                if bn in train_names and (val_count + cnt) <= int(target_val * 1.2):
                    train_names.remove(bn)
                    val_names.append(bn)
                    val_count += cnt
                if val_count >= target_val:
                    break
        train_names = sorted(train_names)
        val_names = sorted(val_names)
        print(f"  最终 训练: {len(train_names)} 图, 验证: {len(val_names)} 图")

    # 阶段3：按划分落盘保存（验证集复制为测试集）
    print("阶段3：落盘保存...")
    train_tiles, val_tiles = save_tiles_for_split(
        args.input_dir, args.output_dir, images_meta, train_names, val_names, tile_size, create_test_folder
    )

    print("\n处理完成！")
    print(f"训练集: {train_tiles} 个小块 (来自 {len(train_names)} 幅原图)")
    print(f"验证集: {val_tiles} 个小块 (来自 {len(val_names)} 幅原图)")
    if create_test_folder:
        print(f"测试集: {val_tiles} 个小块 (与验证集相同)")

    if args.verify:
        print("\n验证输出数据集结构:")
        verify_dataset_structure(args.output_dir)


if __name__ == "__main__":
    main() 