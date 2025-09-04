"""
基于元数据的地理坐标图像合并与数据集划分脚本

功能：
1. 收集图像元数据，计算全局合并边界
2. 基于元数据计算所有切片位置和源图像映射
3. 进行前景平衡的数据集划分
4. 仅在最终输出时处理实际图像数据
"""

import os
import glob
import random
import gc
from typing import Dict, List, Tuple, Optional, NamedTuple
import numpy as np
from PIL import Image
from tqdm import tqdm
import rasterio
from rasterio.warp import reproject, Resampling
from dataclasses import dataclass

# 默认参数
DEFAULT_INPUT_DIR = r"C:\1DataSets\241120\Compare\Datas\Final"
DEFAULT_OUTPUT_DIR = r"C:\1DataSets\241120\Compare\Datas\Split22"
DEFAULT_TILE_SIZE = 512
DEFAULT_VAL_RATIO = 0.2
DEFAULT_BLACK_THRESHOLD = 0.95
DEFAULT_OVERLAP_STRATEGY = "average" # latest, average, max


@dataclass
class ImageMetadata:
    """图像元数据"""
    base_name: str
    files: Dict[str, str]  # 类型 -> 文件路径
    bounds: Tuple[float, float, float, float]  # 地理边界
    crs: str  # 坐标系
    size: Tuple[int, int]  # 像素尺寸
    transform: object  # 地理变换矩阵
    resolution: float  # 分辨率


@dataclass
class TileMetadata:
    """切片元数据"""
    row: int
    col: int
    global_bounds: Tuple[float, float, float, float]  # 全局坐标
    pixel_bounds: Tuple[int, int, int, int]  # 在合并图像中的像素坐标
    source_mappings: List[Tuple[str, Tuple[int, int, int, int]]]  # (base_name, 在源图像中的像素区域)
    foreground_ratio: Optional[float] = None


class MetadataProcessor:
    """元数据处理器"""
    
    def __init__(self, input_dir: str, tile_size: int = 512, 
                 black_threshold: float = 0.95, overlap_strategy: str = "latest"):
        self.input_dir = input_dir
        self.tile_size = tile_size
        self.black_threshold = black_threshold
        self.overlap_strategy = overlap_strategy
        self.images_metadata: List[ImageMetadata] = []
        self.global_bounds: Tuple[float, float, float, float] = None
        self.global_size: Tuple[int, int] = None
        self.global_transform: object = None
        self.target_crs: str = None
        self.target_resolution: float = None
    
    def collect_image_metadata(self):
        """收集所有图像的元数据"""
        print("收集图像元数据...")
        
        # 查找所有图像组
        base_names = set()
        a_files = glob.glob(os.path.join(self.input_dir, "*_A.tif"))
        for a_file in a_files:
            filename = os.path.basename(a_file)
            base_name = filename[:-6] if filename.endswith("_A.tif") else filename
            base_names.add(base_name)
        
        if not base_names:
            raise ValueError("未找到任何符合格式的图像文件")
        
        print(f"找到 {len(base_names)} 个图像组")
        
        # 收集每个图像组的元数据
        for base_name in tqdm(base_names, desc="收集元数据"):
            try:
                # 构建文件路径
                files = {
                    'A': os.path.join(self.input_dir, f"{base_name}_A.tif"),
                    'B': os.path.join(self.input_dir, f"{base_name}_B.tif"),
                    'D': os.path.join(self.input_dir, f"{base_name}_D.tif"),
                    'label': os.path.join(self.input_dir, f"{base_name}_E.png")
                }
                
                # 检查文件存在性
                if not all(os.path.exists(f) for f in files.values()):
                    print(f"警告: 图像组 {base_name} 文件不完整，跳过")
                    continue
                
                # 读取主要TIF文件的元数据
                with rasterio.open(files['A']) as src:
                    bounds = src.bounds
                    crs = str(src.crs)
                    size = (src.width, src.height)
                    transform = src.transform
                    resolution = abs(transform.a)  # x方向分辨率
                
                metadata = ImageMetadata(
                    base_name=base_name,
                    files=files,
                    bounds=bounds,
                    crs=crs,
                    size=size,
                    transform=transform,
                    resolution=resolution
                )
                
                self.images_metadata.append(metadata)
                
            except Exception as e:
                print(f"警告: 处理图像 {base_name} 时出错: {e}")
                continue
        
        if not self.images_metadata:
            raise ValueError("未收集到任何有效的图像元数据")
        
        print(f"成功收集 {len(self.images_metadata)} 个图像的元数据")
    
    def calculate_global_bounds(self):
        """计算全局边界和目标参数"""
        print("计算全局边界...")
        
        # 统计坐标系和分辨率
        crs_counts = {}
        resolutions = []
        
        for metadata in self.images_metadata:
            crs_counts[metadata.crs] = crs_counts.get(metadata.crs, 0) + 1
            resolutions.append(metadata.resolution)
        
        # 选择最常见的坐标系和中位数分辨率
        self.target_crs = max(crs_counts.items(), key=lambda x: x[1])[0]
        self.target_resolution = np.median(resolutions)
        
        print(f"目标坐标系: {self.target_crs}")
        print(f"目标分辨率: {self.target_resolution}")
        
        # 计算全局边界
        min_x = min(m.bounds[0] for m in self.images_metadata)
        min_y = min(m.bounds[1] for m in self.images_metadata)
        max_x = max(m.bounds[2] for m in self.images_metadata)
        max_y = max(m.bounds[3] for m in self.images_metadata)
        
        self.global_bounds = (min_x, min_y, max_x, max_y)
        
        # 计算全局图像尺寸
        width = int((max_x - min_x) / self.target_resolution)
        height = int((max_y - min_y) / self.target_resolution)
        self.global_size = (width, height)
        
        # 创建全局变换矩阵
        self.global_transform = rasterio.transform.from_bounds(
            min_x, min_y, max_x, max_y, width, height
        )
        
        print(f"全局边界: {self.global_bounds}")
        print(f"全局尺寸: {width} x {height}")
    
    def generate_tile_metadata(self) -> List[TileMetadata]:
        """生成所有切片的元数据"""
        print("生成切片元数据...")
        
        width, height = self.global_size
        min_x, min_y, max_x, max_y = self.global_bounds
        
        # 计算切片网格
        rows = height // self.tile_size
        cols = width // self.tile_size
        
        print(f"将生成 {cols} x {rows} = {cols * rows} 个切片")
        
        tiles_metadata = []
        
        for row in tqdm(range(rows), desc="计算切片映射"):
            for col in range(cols):
                # 计算切片在全局图像中的像素边界
                x1 = col * self.tile_size
                y1 = row * self.tile_size
                x2 = x1 + self.tile_size
                y2 = y1 + self.tile_size
                pixel_bounds = (x1, y1, x2, y2)
                
                # 计算切片的地理边界
                tile_min_x = min_x + x1 * self.target_resolution
                tile_max_x = min_x + x2 * self.target_resolution
                tile_max_y = max_y - y1 * self.target_resolution
                tile_min_y = max_y - y2 * self.target_resolution
                global_bounds = (tile_min_x, tile_min_y, tile_max_x, tile_max_y)
                
                # 找到与此切片相交的所有源图像
                source_mappings = []
                for img_meta in self.images_metadata:
                    # 检查地理边界是否相交
                    if (img_meta.bounds[2] > tile_min_x and img_meta.bounds[0] < tile_max_x and
                        img_meta.bounds[3] > tile_min_y and img_meta.bounds[1] < tile_max_y):
                        
                        # 计算在源图像中的像素区域
                        img_transform = img_meta.transform
                        
                        # 将地理坐标转换为源图像的像素坐标
                        src_x1 = max(0, int((tile_min_x - img_meta.bounds[0]) / abs(img_transform.a)))
                        src_y1 = max(0, int((img_meta.bounds[3] - tile_max_y) / abs(img_transform.e)))
                        src_x2 = min(img_meta.size[0], int((tile_max_x - img_meta.bounds[0]) / abs(img_transform.a)))
                        src_y2 = min(img_meta.size[1], int((img_meta.bounds[3] - tile_min_y) / abs(img_transform.e)))
                        
                        if src_x1 < src_x2 and src_y1 < src_y2:
                            source_mappings.append((img_meta.base_name, (src_x1, src_y1, src_x2, src_y2)))
                
                # 只保留有源图像映射的切片
                if source_mappings:
                    tile_meta = TileMetadata(
                        row=row,
                        col=col,
                        global_bounds=global_bounds,
                        pixel_bounds=pixel_bounds,
                        source_mappings=source_mappings
                    )
                    tiles_metadata.append(tile_meta)
        
        print(f"生成了 {len(tiles_metadata)} 个有效切片的元数据")
        return tiles_metadata
    
    def calculate_foreground_ratios(self, tiles_metadata: List[TileMetadata]) -> List[TileMetadata]:
        """计算每个切片的前景比例（仅处理标签）"""
        print("计算切片前景比例...")
        
        for tile_meta in tqdm(tiles_metadata, desc="分析前景比例"):
            try:
                # 创建空的标签tile
                label_tile = np.zeros((self.tile_size, self.tile_size), dtype=np.uint8)
                
                # 合并所有源图像的标签
                for base_name, src_bounds in tile_meta.source_mappings:
                    # 找到对应的图像元数据
                    img_meta = next(m for m in self.images_metadata if m.base_name == base_name)
                    
                    # 读取标签文件
                    label_path = img_meta.files['label']
                    with Image.open(label_path).convert('L') as img:
                        img_array = np.array(img)
                    
                    # 裁剪源区域
                    src_x1, src_y1, src_x2, src_y2 = src_bounds
                    src_region = img_array[src_y1:src_y2, src_x1:src_x2]
                    
                    if src_region.size == 0:
                        continue
                    
                    # 计算在目标tile中的位置
                    # 这里简化处理，假设完全对应
                    dst_h, dst_w = src_region.shape
                    if dst_h <= self.tile_size and dst_w <= self.tile_size:
                        # 使用最大值合并策略
                        label_tile[:dst_h, :dst_w] = np.maximum(
                            label_tile[:dst_h, :dst_w], src_region
                        )
                
                # 计算前景比例
                total_pixels = label_tile.size
                fg_pixels = np.sum(label_tile > 0)
                fg_ratio = fg_pixels / total_pixels if total_pixels > 0 else 0
                
                tile_meta.foreground_ratio = fg_ratio
                
            except Exception as e:
                print(f"警告: 计算切片 ({tile_meta.row}, {tile_meta.col}) 前景比例时出错: {e}")
                tile_meta.foreground_ratio = 0
        
        return tiles_metadata


class DatasetSplitter:
    """数据集划分器"""
    
    def __init__(self, val_ratio: float = 0.2, black_threshold: float = 0.95):
        self.val_ratio = val_ratio
        self.black_threshold = black_threshold
    
    def is_black_tile_from_sources(self, processor: MetadataProcessor, 
                                  tile_meta: TileMetadata) -> bool:
        """基于源图像判断是否为黑色切片"""
        try:
            # 创建A通道的合并tile用于黑块检测
            a_tile = np.zeros((processor.tile_size, processor.tile_size, 3), dtype=np.uint8)
            
            for base_name, src_bounds in tile_meta.source_mappings[:1]:  # 只检查第一个源
                img_meta = next(m for m in processor.images_metadata if m.base_name == base_name)
                
                # 读取A通道文件
                with rasterio.open(img_meta.files['A']) as src:
                    # 读取源区域
                    src_x1, src_y1, src_x2, src_y2 = src_bounds
                    window = rasterio.windows.Window(src_x1, src_y1, src_x2-src_x1, src_y2-src_y1)
                    data = src.read(window=window)  # 读取所有波段
                    
                    # 转换为 (H, W, C) 格式
                    if data.ndim == 3:
                        data = np.transpose(data, (1, 2, 0))
                    
                    # 调整大小以适应tile
                    if data.shape[:2] != (processor.tile_size, processor.tile_size):
                        img = Image.fromarray(data.astype(np.uint8))
                        img = img.resize((processor.tile_size, processor.tile_size), Image.NEAREST)
                        data = np.array(img)
                    
                    # 使用最新策略合并
                    valid_mask = np.any(data > 0, axis=-1)
                    a_tile[valid_mask] = data[valid_mask]
                    break  # 只检查第一个有效源
            
            # 检查是否为黑块
            black_pixels = np.sum(np.all(a_tile <= 5, axis=-1))
            total_pixels = a_tile.shape[0] * a_tile.shape[1]
            ratio = black_pixels / total_pixels if total_pixels > 0 else 0
            
            return ratio >= self.black_threshold
            
        except Exception as e:
            print(f"警告: 检测黑块时出错: {e}")
            return False
    
    def split_tiles(self, processor: MetadataProcessor, 
                   tiles_metadata: List[TileMetadata]) -> Tuple[List[TileMetadata], List[TileMetadata]]:
        """划分训练集和验证集"""
        print("划分数据集...")
        
        # 过滤黑色切片
        print("过滤黑色切片...")
        valid_tiles = []
        for tile_meta in tqdm(tiles_metadata, desc="过滤黑块"):
            if not self.is_black_tile_from_sources(processor, tile_meta):
                valid_tiles.append(tile_meta)
        
        print(f"过滤后保留 {len(valid_tiles)} 个有效切片")
        
        if not valid_tiles:
            raise ValueError("没有有效的切片")
        
        # 计算全局前景比例
        fg_ratios = [t.foreground_ratio or 0 for t in valid_tiles]
        global_fg_ratio = np.mean(fg_ratios)
        print(f"全局前景比例: {global_fg_ratio:.4f}")
        
        # 使用分层抽样确保前景比例平衡
        train_tiles, val_tiles = self._stratified_split(valid_tiles)
        
        # 计算初步前景比例
        train_fg_ratio = np.mean([t.foreground_ratio or 0 for t in train_tiles])
        val_fg_ratio = np.mean([t.foreground_ratio or 0 for t in val_tiles])
        
        print(f"分层抽样后:")
        print(f"  训练集: {len(train_tiles)} 个切片，前景比例: {train_fg_ratio:.4f}")
        print(f"  验证集: {len(val_tiles)} 个切片，前景比例: {val_fg_ratio:.4f}")
        print(f"  前景比例差异: {abs(train_fg_ratio - val_fg_ratio):.4f}")
        
        # 如果差异过大，进行微调
        ratio_diff = abs(train_fg_ratio - val_fg_ratio)
        if ratio_diff > 0.02:  # 差异超过2%时进行调整
            print("前景比例差异较大，进行微调...")
            train_tiles, val_tiles = self._fine_tune_balance(train_tiles, val_tiles, global_fg_ratio)
            
            # 重新计算
            train_fg_ratio = np.mean([t.foreground_ratio or 0 for t in train_tiles])
            val_fg_ratio = np.mean([t.foreground_ratio or 0 for t in val_tiles])
            
            print(f"微调后:")
            print(f"  训练集: {len(train_tiles)} 个切片，前景比例: {train_fg_ratio:.4f}")
            print(f"  验证集: {len(val_tiles)} 个切片，前景比例: {val_fg_ratio:.4f}")
            print(f"  前景比例差异: {abs(train_fg_ratio - val_fg_ratio):.4f}")
        
        # 显示详细的平衡度分析
        self._analyze_balance(train_tiles, val_tiles, global_fg_ratio)
        
        return train_tiles, val_tiles
    
    def _analyze_balance(self, train_tiles: List[TileMetadata], 
                        val_tiles: List[TileMetadata], global_fg_ratio: float):
        """分析数据集平衡度"""
        print("\n=== 数据集平衡度分析 ===")
        
        train_ratios = [t.foreground_ratio or 0 for t in train_tiles]
        val_ratios = [t.foreground_ratio or 0 for t in val_tiles]
        
        # 基础统计
        train_mean = np.mean(train_ratios)
        val_mean = np.mean(val_ratios)
        train_std = np.std(train_ratios)
        val_std = np.std(val_ratios)
        
        print(f"全局前景比例: {global_fg_ratio:.4f}")
        print(f"训练集统计: 均值={train_mean:.4f}, 标准差={train_std:.4f}")
        print(f"验证集统计: 均值={val_mean:.4f}, 标准差={val_std:.4f}")
        print(f"均值差异: {abs(train_mean - val_mean):.4f}")
        
        # 分布分析
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
        train_hist, _ = np.histogram(train_ratios, bins=bins)
        val_hist, _ = np.histogram(val_ratios, bins=bins)
        
        print("\n前景比例分布:")
        print("区间范围       训练集    验证集    比例差异")
        print("-" * 40)
        
        for i in range(len(bins)-1):
            train_pct = train_hist[i] / len(train_tiles) * 100 if len(train_tiles) > 0 else 0
            val_pct = val_hist[i] / len(val_tiles) * 100 if len(val_tiles) > 0 else 0
            diff = abs(train_pct - val_pct)
            
            print(f"[{bins[i]:.1f}-{bins[i+1]:.1f})    {train_pct:6.1f}%  {val_pct:6.1f}%  {diff:6.1f}%")
        
        # 平衡度评分
        ratio_diff = abs(train_mean - val_mean)
        if ratio_diff < 0.01:
            balance_score = "优秀"
        elif ratio_diff < 0.02:
            balance_score = "良好"
        elif ratio_diff < 0.05:
            balance_score = "一般"
        else:
            balance_score = "需要改进"
            
        print(f"\n平衡度评分: {balance_score}")
        print("=" * 30)
    
    def _stratified_split(self, tiles: List[TileMetadata]) -> Tuple[List[TileMetadata], List[TileMetadata]]:
        """分层抽样划分数据集"""
        import random
        
        # 随机打乱避免位置偏差
        shuffled_tiles = tiles.copy()
        random.shuffle(shuffled_tiles)
        
        # 按前景比例分成多个层
        n_strata = min(10, len(tiles) // 20)  # 最多10层，每层至少20个样本
        if n_strata < 2:
            n_strata = 2
        
        # 按前景比例排序
        sorted_tiles = sorted(shuffled_tiles, key=lambda t: t.foreground_ratio or 0)
        
        # 分层
        strata = []
        stratum_size = len(sorted_tiles) // n_strata
        
        for i in range(n_strata):
            start_idx = i * stratum_size
            if i == n_strata - 1:  # 最后一层包含所有剩余样本
                end_idx = len(sorted_tiles)
            else:
                end_idx = (i + 1) * stratum_size
            
            stratum = sorted_tiles[start_idx:end_idx]
            if stratum:  # 确保层不为空
                strata.append(stratum)
        
        print(f"使用 {len(strata)} 个层进行分层抽样")
        
        # 在每层中按比例抽样
        train_tiles = []
        val_tiles = []
        
        for i, stratum in enumerate(strata):
            # 计算这一层应该分配给验证集的样本数
            stratum_val_count = max(1, int(len(stratum) * self.val_ratio))
            
            # 随机抽样
            random.shuffle(stratum)
            stratum_val = stratum[:stratum_val_count]
            stratum_train = stratum[stratum_val_count:]
            
            val_tiles.extend(stratum_val)
            train_tiles.extend(stratum_train)
            
            # 输出每层的统计信息
            stratum_fg_ratio = np.mean([t.foreground_ratio or 0 for t in stratum])
            stratum_val_fg_ratio = np.mean([t.foreground_ratio or 0 for t in stratum_val]) if stratum_val else 0
            stratum_train_fg_ratio = np.mean([t.foreground_ratio or 0 for t in stratum_train]) if stratum_train else 0
            
            print(f"  层 {i+1}: 总数={len(stratum)}, 训练={len(stratum_train)}, 验证={len(stratum_val)}")
            print(f"       前景比例: 层均值={stratum_fg_ratio:.4f}, 训练={stratum_train_fg_ratio:.4f}, 验证={stratum_val_fg_ratio:.4f}")
        
        return train_tiles, val_tiles
    
    def _fine_tune_balance(self, train_tiles: List[TileMetadata], 
                          val_tiles: List[TileMetadata], 
                          target_ratio: float) -> Tuple[List[TileMetadata], List[TileMetadata]]:
        """微调训练集和验证集的前景比例平衡"""
        import random
        
        # 计算当前比例
        train_fg_ratio = np.mean([t.foreground_ratio or 0 for t in train_tiles])
        val_fg_ratio = np.mean([t.foreground_ratio or 0 for t in val_tiles])
        
        # 确定需要调整的方向
        max_swaps = min(len(train_tiles), len(val_tiles)) // 20  # 最多交换5%的样本
        
        for _ in range(max_swaps):
            # 重新计算当前比例
            train_fg_ratio = np.mean([t.foreground_ratio or 0 for t in train_tiles])
            val_fg_ratio = np.mean([t.foreground_ratio or 0 for t in val_tiles])
            
            # 如果已经足够平衡，停止调整
            if abs(train_fg_ratio - val_fg_ratio) < 0.01:
                break
            
            if train_fg_ratio > val_fg_ratio:
                # 训练集前景比例过高，需要从训练集找高前景比例样本给验证集
                # 从验证集找低前景比例样本给训练集
                
                # 找到训练集中前景比例最高的样本
                train_high = max(train_tiles, key=lambda t: t.foreground_ratio or 0)
                # 找到验证集中前景比例最低的样本
                val_low = min(val_tiles, key=lambda t: t.foreground_ratio or 0)
                
                # 如果交换能改善平衡，则交换
                if (train_high.foreground_ratio or 0) > (val_low.foreground_ratio or 0):
                    train_tiles.remove(train_high)
                    val_tiles.remove(val_low)
                    train_tiles.append(val_low)
                    val_tiles.append(train_high)
                else:
                    break
                    
            else:
                # 验证集前景比例过高
                val_high = max(val_tiles, key=lambda t: t.foreground_ratio or 0)
                train_low = min(train_tiles, key=lambda t: t.foreground_ratio or 0)
                
                if (val_high.foreground_ratio or 0) > (train_low.foreground_ratio or 0):
                    val_tiles.remove(val_high)
                    train_tiles.remove(train_low)
                    val_tiles.append(train_low)
                    train_tiles.append(val_high)
                else:
                    break
        
        return train_tiles, val_tiles


class TileGenerator:
    """切片生成器 - 只在输出时处理实际图像"""
    
    def __init__(self, processor: MetadataProcessor):
        self.processor = processor
    
    def generate_tile(self, tile_meta: TileMetadata) -> Dict[str, np.ndarray]:
        """生成单个切片的实际图像数据"""
        tile_data = {}
        
        # 为每种类型创建空tile
        for img_type in ['A', 'B', 'D', 'label']:
            if img_type == 'label':
                tile_data[img_type] = np.zeros((self.processor.tile_size, self.processor.tile_size), dtype=np.uint8)
            else:
                # 假设3波段
                tile_data[img_type] = np.zeros((self.processor.tile_size, self.processor.tile_size, 3), dtype=np.uint8)
        
        # 合并所有源图像
        for base_name, src_bounds in tile_meta.source_mappings:
            img_meta = next(m for m in self.processor.images_metadata if m.base_name == base_name)
            src_x1, src_y1, src_x2, src_y2 = src_bounds
            
            # 处理每种图像类型
            for img_type in ['A', 'B', 'D', 'label']:
                try:
                    file_path = img_meta.files[img_type]
                    
                    if img_type == 'label':
                        # PNG标签文件
                        with Image.open(file_path).convert('L') as img:
                            img_array = np.array(img)
                        
                        src_region = img_array[src_y1:src_y2, src_x1:src_x2]
                        
                        if src_region.size > 0:
                            # 调整大小
                            if src_region.shape != (self.processor.tile_size, self.processor.tile_size):
                                img_pil = Image.fromarray(src_region)
                                img_pil = img_pil.resize((self.processor.tile_size, self.processor.tile_size), Image.NEAREST)
                                src_region = np.array(img_pil)
                            
                            # 使用最大值合并
                            tile_data[img_type] = np.maximum(tile_data[img_type], src_region)
                    else:
                        # TIF文件
                        with rasterio.open(file_path) as src:
                            window = rasterio.windows.Window(src_x1, src_y1, src_x2-src_x1, src_y2-src_y1)
                            data = src.read(window=window)
                            
                            if data.ndim == 3:
                                data = np.transpose(data, (1, 2, 0))
                            
                            # 调整大小
                            if data.shape[:2] != (self.processor.tile_size, self.processor.tile_size):
                                img_pil = Image.fromarray(data.astype(np.uint8))
                                img_pil = img_pil.resize((self.processor.tile_size, self.processor.tile_size), Image.NEAREST)
                                data = np.array(img_pil)
                            
                            # 使用latest策略
                            valid_mask = np.any(data > 0, axis=-1)
                            tile_data[img_type][valid_mask] = data[valid_mask]
                
                except Exception as e:
                    print(f"警告: 处理 {base_name} 的 {img_type} 时出错: {e}")
                    continue
        
        return tile_data
    
    def save_dataset(self, train_tiles: List[TileMetadata], 
                    val_tiles: List[TileMetadata], output_dir: str):
        """保存数据集"""
        print("保存数据集...")
        
        # 创建输出目录
        splits = ['train', 'val', 'test']
        types = ['A', 'B', 'C', 'label']
        
        for split in splits:
            for img_type in types:
                os.makedirs(os.path.join(output_dir, split, img_type), exist_ok=True)
        
        # 保存训练集
        self._save_split(train_tiles, output_dir, 'train')
        
        # 保存验证集
        self._save_split(val_tiles, output_dir, 'val')
        
        # 复制验证集为测试集
        self._save_split(val_tiles, output_dir, 'test')
        
        print("数据集保存完成!")
    
    def _save_split(self, tiles: List[TileMetadata], output_dir: str, split: str):
        """保存一个数据集分割"""
        type_mapping = {'A': 'A', 'B': 'B', 'D': 'C', 'label': 'label'}
        
        for tile_meta in tqdm(tiles, desc=f"保存{split}集"):
            # 生成切片数据
            tile_data = self.generate_tile(tile_meta)
            
            # 构建文件名
            base_name = f"tile_r{tile_meta.row}_c{tile_meta.col}"
            
            # 保存每种类型
            for src_type, dst_type in type_mapping.items():
                data = tile_data[src_type]
                
                # 确保数据类型正确
                if data.dtype != np.uint8:
                    if data.max() <= 1.0:
                        data = (data * 255).astype(np.uint8)
                    else:
                        data = data.astype(np.uint8)
                
                # 保存为PNG
                img = Image.fromarray(data)
                save_path = os.path.join(output_dir, split, dst_type, f"{base_name}.png")
                img.save(save_path)


def main():
    """主函数"""
    print("开始基于元数据的地理图像合并与数据集划分...")
    
    # 设置随机种子确保可重现性
    import random
    random.seed(42)
    np.random.seed(42)
    
    # 使用默认参数
    input_dir = DEFAULT_INPUT_DIR
    output_dir = DEFAULT_OUTPUT_DIR
    tile_size = DEFAULT_TILE_SIZE
    val_ratio = DEFAULT_VAL_RATIO
    black_threshold = DEFAULT_BLACK_THRESHOLD
    
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"切片大小: {tile_size}x{tile_size}")
    print(f"验证集比例: {val_ratio}")
    print(f"重叠处理策略: {DEFAULT_OVERLAP_STRATEGY}")
    print(f"随机种子: 42 (确保结果可重现)")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 第一步：收集图像元数据
        processor = MetadataProcessor(input_dir, tile_size, black_threshold)
        processor.collect_image_metadata()
        processor.calculate_global_bounds()
        
        # 第二步：生成切片元数据
        tiles_metadata = processor.generate_tile_metadata()
        
        # 第三步：计算前景比例
        tiles_metadata = processor.calculate_foreground_ratios(tiles_metadata)
        
        # 第四步：划分数据集
        splitter = DatasetSplitter(val_ratio, black_threshold)
        train_tiles, val_tiles = splitter.split_tiles(processor, tiles_metadata)
        
        # 第五步：生成并保存实际图像
        generator = TileGenerator(processor)
        generator.save_dataset(train_tiles, val_tiles, output_dir)
        
        print(f"\n处理完成!")
        print(f"训练集: {len(train_tiles)} 个切片")
        print(f"验证集: {len(val_tiles)} 个切片")
        print(f"测试集: {len(val_tiles)} 个切片 (与验证集相同)")
        
    except Exception as e:
        print(f"错误: {e}")
        raise


if __name__ == "__main__":
    main()
