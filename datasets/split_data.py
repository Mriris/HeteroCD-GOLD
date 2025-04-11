import argparse
import glob
import os
import random
import re
import shutil
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm

# 默认参数设置（放在代码最上方以便于修改）
DEFAULT_INPUT_DIR = "/data/jingwei/yantingxuan/Datasets/CityCN/Enhanced"
DEFAULT_OUTPUT_DIR = "/data/jingwei/yantingxuan/Datasets/CityCN/Split5"
DEFAULT_VERIFY = False  # 是否默认验证数据集完整性
DEFAULT_VAL_RATIO = 0.2  # 验证集占总数据的比例
DEFAULT_TEST_FOLDER = True  # 是否创建测试集文件夹（内容与验证集相同）
DEFAULT_CHECK_FORMAT = False  # 是否检查图像格式
DEFAULT_SORT_BY_CHANGE_AREA = False  # 是否根据变化区域大小进行排序选择
DEFAULT_SELECT_PERCENTAGE = 100.0  # 默认选择所有图像


def check_image_format(img_path, required_mode=None, required_bands=None):
    """
    检查图像格式是否符合要求，并提供详细的格式信息

    参数:
        img_path: 图像路径
        required_mode: 要求的图像模式，如'L'为灰度，'RGB'为彩色
        required_bands: 要求的波段数

    返回:
        (是否通过检查, 错误信息, 格式详情)
    """
    try:
        with Image.open(img_path) as img:
            # 获取图像的基本信息
            img_mode = img.mode
            img_size = img.size

            # 将图像转换为numpy数组以获取更多信息
            img_array = np.array(img)

            # 获取通道数
            if len(img_array.shape) == 2:
                actual_bands = 1
                shape_info = f"{img_array.shape[0]}x{img_array.shape[1]}"
            else:
                actual_bands = img_array.shape[2]
                shape_info = f"{img_array.shape[0]}x{img_array.shape[1]}x{img_array.shape[2]}"

            # 获取数据类型
            data_type = img_array.dtype

            # 获取像素值范围
            min_val = img_array.min()
            max_val = img_array.max()

            # 对于灰度图像，获取唯一值
            if actual_bands == 1:
                unique_values = np.unique(img_array)
                num_unique = len(unique_values)
                unique_info = f"唯一值数量: {num_unique}"
                if num_unique <= 10:  # 如果唯一值较少，列出所有值
                    unique_info += f", 值: {sorted(unique_values.tolist())}"
            else:
                unique_info = "多通道图像，略过唯一值统计"

            # 构建格式详情
            format_details = (
                f"模式: {img_mode}, 尺寸: {img_size}, 形状: {shape_info}, "
                f"数据类型: {data_type}, 值范围: [{min_val}, {max_val}], {unique_info}"
            )

            # 检查模式是否符合要求
            if required_mode and img_mode != required_mode:
                return False, f"图像模式错误: 需要{required_mode}模式，实际为{img_mode}模式", format_details

            # 检查波段数是否符合要求
            if required_bands and actual_bands != required_bands:
                return False, f"图像波段错误: 需要{required_bands}波段，实际为{actual_bands}波段", format_details

            # 如果是要求灰度二值图，检查是否真的只有两个值
            if required_mode == 'L' and num_unique != 2 and not (
                    num_unique == 1 and (unique_values[0] == 0 or unique_values[0] == 255)):
                return False, f"二值图检查失败: 图像既不是严格的二值图，也不是纯黑或纯白图像", format_details

            return True, "检查通过", format_details

    except Exception as e:
        return False, f"图像检查异常: {str(e)}", "无法获取格式详情"


def analyze_label_images(files, sample_size=None):
    """
    分析标签图像的格式特征

    参数:
        files: 标签图像文件列表
        sample_size: 抽样分析的图像数量，None表示分析所有图像

    返回:
        分析结果描述
    """
    if sample_size:
        files_to_analyze = random.sample(files, min(sample_size, len(files)))
    else:
        files_to_analyze = files

    modes = {}
    unique_value_counts = {}
    value_ranges = {}

    for file_path in tqdm(files_to_analyze, desc="分析标签图像特征"):
        try:
            with Image.open(file_path) as img:
                # 记录模式
                mode = img.mode
                if mode in modes:
                    modes[mode] += 1
                else:
                    modes[mode] = 1

                # 分析像素值
                img_array = np.array(img)
                unique_vals = np.unique(img_array)
                num_unique = len(unique_vals)

                # 记录唯一值数量
                if num_unique in unique_value_counts:
                    unique_value_counts[num_unique] += 1
                else:
                    unique_value_counts[num_unique] = 1

                # 记录值范围
                min_val = img_array.min()
                max_val = img_array.max()
                range_key = f"{min_val}-{max_val}"
                if range_key in value_ranges:
                    value_ranges[range_key] += 1
                else:
                    value_ranges[range_key] = 1
        except Exception as e:
            print(f"分析文件 {os.path.basename(file_path)} 时出错: {e}")

    # 生成分析报告
    report = []

    report.append(f"分析了 {len(files_to_analyze)} 个标签图像:")

    report.append("\n图像模式分布:")
    for mode, count in modes.items():
        percentage = (count / len(files_to_analyze)) * 100
        report.append(f"  - {mode}: {count} 个 ({percentage:.1f}%)")

    report.append("\n唯一值数量分布:")
    for num, count in sorted(unique_value_counts.items()):
        percentage = (count / len(files_to_analyze)) * 100
        report.append(f"  - {num} 个唯一值: {count} 个图像 ({percentage:.1f}%)")

    report.append("\n像素值范围分布:")
    for range_key, count in sorted(value_ranges.items()):
        percentage = (count / len(files_to_analyze)) * 100
        report.append(f"  - 范围 {range_key}: {count} 个图像 ({percentage:.1f}%)")

    return "\n".join(report)


def calculate_change_area(label_path):
    """
    计算标签图像中白色区域（变化区域）的像素数量

    参数:
        label_path: 标签图像路径

    返回:
        白色像素的数量
    """
    try:
        with Image.open(label_path) as img:
            img_array = np.array(img)
            # 假设白色为255，表示变化区域
            white_pixels = np.sum(img_array == 255)
            return white_pixels
    except Exception as e:
        print(f"计算变化区域时出错: {e}")
        return 0


def extract_original_base_name(filename):
    """
    从预处理后的文件名中提取原始图像名称
    预处理后的文件格式为：{base_name}_{aug_type}_x{x}_y{y}_{A/B/D/E}.{ext}
    
    参数:
        filename: 文件名（含扩展名）
    
    返回:
        原始图像基本名称
    """
    # 移除文件扩展名和A/B/D/E标识
    base_without_ext = os.path.splitext(filename)[0]
    if base_without_ext.endswith("_A") or base_without_ext.endswith("_B") or \
            base_without_ext.endswith("_D") or base_without_ext.endswith("_E"):
        base_without_ext = base_without_ext[:-2]

    # 使用正则表达式匹配原始名称
    # 这里假设原始名称后面跟着_original、_h_flip等增强类型标识和坐标信息
    match = re.match(r'(.+?)_(original|h_flip|v_flip|rot90|rot180|rot270)_x\d+_y\d+', base_without_ext)
    if match:
        return match.group(1)

    # 如果不匹配上述格式，可能是其他格式或原始文件，直接返回
    return base_without_ext


def reorganize_processed_images(input_dir, output_dir, val_ratio=0.2, create_test_folder=True, check_format=True,
                                sort_by_change_area=False, select_percentage=100.0):
    """
    重新组织处理后的图像文件，确保同一原始图像的所有增强版本都分配到同一数据集
    基于生成小块数量而非原始图像数量分割数据集，以更准确地达到指定的验证集比例
    
    参数:
        input_dir: 处理后图像所在的输入目录
        output_dir: 重组后的输出目录基础路径
        val_ratio: 验证集占总数据的比例
        create_test_folder: 是否创建测试集文件夹（内容与验证集相同）
        check_format: 是否检查图像格式
        sort_by_change_area: 是否根据变化区域大小进行排序选择
        select_percentage: 选择的图像百分比（0-100）
    """
    # 创建输出目录结构
    train_folder = os.path.join(output_dir, "train")
    val_folder = os.path.join(output_dir, "val")
    test_folder = os.path.join(output_dir, "test") if create_test_folder else None

    folder_A_train = os.path.join(train_folder, "A")
    folder_B_train = os.path.join(train_folder, "B")
    folder_C_train = os.path.join(train_folder, "C")
    folder_label_train = os.path.join(train_folder, "label")

    folder_A_val = os.path.join(val_folder, "A")
    folder_B_val = os.path.join(val_folder, "B")
    folder_C_val = os.path.join(val_folder, "C")
    folder_label_val = os.path.join(val_folder, "label")

    # 创建文件夹
    folders = [folder_A_train, folder_B_train, folder_C_train, folder_label_train,
               folder_A_val, folder_B_val, folder_C_val, folder_label_val]

    if create_test_folder:
        folder_A_test = os.path.join(test_folder, "A")
        folder_B_test = os.path.join(test_folder, "B")
        folder_C_test = os.path.join(test_folder, "C")
        folder_label_test = os.path.join(test_folder, "label")
        folders.extend([folder_A_test, folder_B_test, folder_C_test, folder_label_test])

    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    # 查找所有相关文件
    files_A = glob.glob(os.path.join(input_dir, "*_A.tif"))
    files_B = glob.glob(os.path.join(input_dir, "*_B.tif"))
    files_C = glob.glob(os.path.join(input_dir, "*_D.tif"))  # 修改为 *_D.tif
    files_E = glob.glob(os.path.join(input_dir, "*_E.png"))

    print(f"找到 {len(files_A)} 个 A 类图像")
    print(f"找到 {len(files_B)} 个 B 类图像")
    print(f"找到 {len(files_C)} 个 C 类图像")
    print(f"找到 {len(files_E)} 个标签图像")

    # 检查格式
    if check_format:
        print("检查图像格式...")
        format_errors = []
        format_details = []

        # 检查A类图像
        for file_path in tqdm(files_A[:min(10, len(files_A))], desc="检查 A 类图像格式"):
            passed, error_msg, details = check_image_format(file_path, required_bands=3)
            if not passed:
                format_errors.append(f"A类图像 {os.path.basename(file_path)}: {error_msg}")
                format_details.append(f"A类图像 {os.path.basename(file_path)}: {details}")

        # 检查B类图像
        for file_path in tqdm(files_B[:min(10, len(files_B))], desc="检查 B 类图像格式"):
            passed, error_msg, details = check_image_format(file_path, required_bands=3)
            if not passed:
                format_errors.append(f"B类图像 {os.path.basename(file_path)}: {error_msg}")
                format_details.append(f"B类图像 {os.path.basename(file_path)}: {details}")

        # 检查C类图像
        for file_path in tqdm(files_C[:min(10, len(files_C))], desc="检查 C 类图像格式"):
            passed, error_msg, details = check_image_format(file_path, required_bands=3)
            if not passed:
                format_errors.append(f"C类图像 {os.path.basename(file_path)}: {error_msg}")
                format_details.append(f"C类图像 {os.path.basename(file_path)}: {details}")

        # 检查标签图像
        label_format_issues = []
        for file_path in tqdm(files_E[:min(10, len(files_E))], desc="检查标签图像格式"):
            passed, error_msg, details = check_image_format(file_path, required_mode='L')
            if not passed:
                format_errors.append(f"标签图像 {os.path.basename(file_path)}: {error_msg}")
                format_details.append(f"标签图像 {os.path.basename(file_path)}: {details}")
                label_format_issues.append(file_path)

        if format_errors:
            print("发现图像格式问题:")
            for error in format_errors[:10]:
                print(f"  - {error}")
            if len(format_errors) > 10:
                print(f"  ... 以及其他 {len(format_errors) - 10} 个问题")

            print("\n图像格式详情:")
            for detail in format_details[:10]:
                print(f"  - {detail}")
            if len(format_details) > 10:
                print(f"  ... 以及其他 {len(format_details) - 10} 个图像的详情")

            # 如果标签图像有问题，进行更深入的分析
            if label_format_issues:
                print("\n对标签图像进行更深入分析...")
                label_analysis = analyze_label_images(files_E, sample_size=min(100, len(files_E)))
                print(label_analysis)

                print("\n问题标签图像示例分析:")
                for i, problem_file in enumerate(label_format_issues[:5]):
                    try:
                        with Image.open(problem_file) as img:
                            img_array = np.array(img)
                            unique_values = np.unique(img_array)
                            print(f"问题图像 {i + 1}: {os.path.basename(problem_file)}")
                            print(f"  模式: {img.mode}, 大小: {img.size}")
                            print(f"  唯一值: {unique_values}")
                            print(f"  值范围: [{img_array.min()}, {img_array.max()}]")
                    except Exception as e:
                        print(f"分析问题图像时出错: {e}")

            continue_anyway = input("检测到图像格式问题，是否继续执行？(y/n): ").lower() == 'y'
            if not continue_anyway:
                print("操作已取消")
                return

    # 提取所有文件的原始基本名称并计算每个原始图像生成的小块数量
    print("提取原始图像名称并统计小块数量...")
    original_base_names = set()
    original_to_chunks = defaultdict(int)  # 用于存储每个原始图像生成的小块数量
    filename_to_original = {}  # 用于将文件名映射到原始图像名
    
    # 使用A类图像统计
    for file_path in files_A:
        filename = os.path.basename(file_path)
        original_name = extract_original_base_name(filename)
        original_base_names.add(original_name)
        original_to_chunks[original_name] += 1
        filename_to_original[filename] = original_name
    
    original_base_names = list(original_base_names)
    total_chunks = sum(original_to_chunks.values())
    
    print(f"发现 {len(original_base_names)} 个原始图像，总共生成 {total_chunks} 个小块")
    
    # 显示小块分布统计
    chunk_counts = sorted(original_to_chunks.values())
    if chunk_counts:
        print(f"每个原始图像生成的小块数量统计:")
        print(f"  最小值: {min(chunk_counts)}")
        print(f"  最大值: {max(chunk_counts)}")
        print(f"  平均值: {sum(chunk_counts) / len(chunk_counts):.2f}")
        print(f"  中位数: {chunk_counts[len(chunk_counts) // 2]}")

    # 如果需要根据变化区域大小排序，使用原始图像计算
    if sort_by_change_area:
        print("根据变化区域大小排序原始图像...")
        area_data = []

        for original_name in tqdm(original_base_names, desc="计算变化区域"):
            # 查找该原始名称对应的第一个标签图像
            matching_labels = []
            for label_path in files_E:
                filename = os.path.basename(label_path)
                if extract_original_base_name(filename) == original_name:
                    matching_labels.append(label_path)
                    break

            if matching_labels:
                # 使用第一个匹配的标签图像计算变化区域
                area = calculate_change_area(matching_labels[0])
                area_data.append((original_name, area, original_to_chunks[original_name]))
            else:
                print(f"警告：未找到原始图像 {original_name} 的标签文件")

        # 根据变化区域大小降序排序
        area_data.sort(key=lambda x: x[1], reverse=True)

        # 选择前 select_percentage% 的图像
        select_count = int(len(area_data) * (select_percentage / 100.0))
        selected_original_names = [data[0] for data in area_data[:select_count]]
        
        # 更新每个原始图像生成的小块数量
        selected_original_to_chunks = {name: original_to_chunks[name] for name in selected_original_names}
        selected_total_chunks = sum(selected_original_to_chunks.values())

        print(f"根据变化区域大小选择了 {len(selected_original_names)} 个原始图像 ({select_percentage}%)，总共 {selected_total_chunks} 个小块")

        # 统计选择的图像的变化区域大小分布
        if selected_original_names:
            areas = [data[1] for data in area_data[:select_count]]
            print(f"选择的图像变化区域大小统计:")
            print(f"  最小值: {min(areas)}")
            print(f"  最大值: {max(areas)}")
            print(f"  平均值: {sum(areas) / len(areas):.2f}")
            print(f"  中位数: {sorted(areas)[len(areas) // 2]}")
    else:
        selected_original_names = original_base_names
        selected_original_to_chunks = original_to_chunks
        selected_total_chunks = total_chunks
        print(f"将处理所有 {len(selected_original_names)} 个原始图像，总共 {selected_total_chunks} 个小块")

    # 基于生成小块数量划分数据集
    print("基于生成小块数量划分数据集...")
    
    # 计算验证集应该含有的小块数量
    target_val_chunks = int(selected_total_chunks * val_ratio)
    print(f"目标验证集小块数量: {target_val_chunks} ({val_ratio * 100:.1f}%)")
    
    # 随机打乱原始图像列表
    random.shuffle(selected_original_names)
    
    # 贪心算法：尽量使验证集的小块数量接近目标值
    val_original_names = []
    train_original_names = []
    
    current_val_chunks = 0
    
    # 先将原始图像按生成小块数量排序（从大到小）
    sorted_originals = sorted(selected_original_names, key=lambda x: selected_original_to_chunks[x], reverse=True)
    
    # 使用贪心算法分配图像
    remaining_chunks = target_val_chunks
    
    # 第一步：优先分配大小最接近剩余所需数量的图像
    for orig_name in sorted(sorted_originals, key=lambda x: abs(selected_original_to_chunks[x] - remaining_chunks)):
        chunks = selected_original_to_chunks[orig_name]
        if current_val_chunks + chunks <= target_val_chunks * 1.1:  # 允许最多超过目标值10%
            val_original_names.append(orig_name)
            current_val_chunks += chunks
            remaining_chunks -= chunks
        else:
            train_original_names.append(orig_name)
    
    # 第二步：如果验证集小块数量不足，从训练集中移动一些图像到验证集
    if current_val_chunks < target_val_chunks * 0.9:  # 如果不足目标值的90%
        # 按小块数量从小到大排序训练集图像
        train_original_names.sort(key=lambda x: selected_original_to_chunks[x])
        
        while train_original_names and current_val_chunks < target_val_chunks * 0.95:
            # 从训练集中取出小块数量最小的图像
            orig_name = train_original_names.pop(0)
            chunks = selected_original_to_chunks[orig_name]
            val_original_names.append(orig_name)
            current_val_chunks += chunks
    
    # 打乱验证集和训练集的顺序（保持随机性）
    random.shuffle(val_original_names)
    random.shuffle(train_original_names)
    
    train_chunks = selected_total_chunks - current_val_chunks
    
    print(f"分配结果:")
    print(f"  训练集: {len(train_original_names)} 个原始图像，{train_chunks} 个小块 ({train_chunks/selected_total_chunks*100:.1f}%)")
    print(f"  验证集: {len(val_original_names)} 个原始图像，{current_val_chunks} 个小块 ({current_val_chunks/selected_total_chunks*100:.1f}%)")

    # 创建原始名称到目标集合的映射
    original_to_target_set = {}
    for name in train_original_names:
        original_to_target_set[name] = "train"
    for name in val_original_names:
        original_to_target_set[name] = "val"

    # 处理所有图像文件
    processed_train = 0
    processed_val = 0

    # 处理A类图像
    print("处理A类图像...")
    for file_path in tqdm(files_A):
        filename = os.path.basename(file_path)
        original_name = extract_original_base_name(filename)
        
        # 跳过不在选择列表中的原始图像
        if original_name not in original_to_target_set:
            continue
            
        target_set = original_to_target_set[original_name]
        new_filename = f"{filename.replace('_A.tif', '')}.png"
        
        if target_set == "train":
            target_path = os.path.join(folder_A_train, new_filename)
            processed_train += 1
        else:  # val
            target_path = os.path.join(folder_A_val, new_filename)
            processed_val += 1
            
        # 转换并保存图像
        try:
            with Image.open(file_path) as img:
                img.save(target_path, "PNG")
                
                # 如果需要创建测试集，也复制到测试集
                if create_test_folder and target_set == "val":
                    test_path = os.path.join(folder_A_test, new_filename)
                    img.save(test_path, "PNG")
        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")
    
    # 处理B类图像
    print("处理B类图像...")
    for file_path in tqdm(files_B):
        filename = os.path.basename(file_path)
        original_name = extract_original_base_name(filename)
        
        # 跳过不在选择列表中的原始图像
        if original_name not in original_to_target_set:
            continue
            
        target_set = original_to_target_set[original_name]
        new_filename = f"{filename.replace('_B.tif', '')}.png"
        
        if target_set == "train":
            target_path = os.path.join(folder_B_train, new_filename)
        else:  # val
            target_path = os.path.join(folder_B_val, new_filename)
            
        # 转换并保存图像
        try:
            with Image.open(file_path) as img:
                img.save(target_path, "PNG")
                
                # 如果需要创建测试集，也复制到测试集
                if create_test_folder and target_set == "val":
                    test_path = os.path.join(folder_B_test, new_filename)
                    img.save(test_path, "PNG")
        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")
    
    # 处理C类图像
    print("处理C类图像...")
    for file_path in tqdm(files_C):
        filename = os.path.basename(file_path)
        original_name = extract_original_base_name(filename)
        
        # 跳过不在选择列表中的原始图像
        if original_name not in original_to_target_set:
            continue
            
        target_set = original_to_target_set[original_name]
        new_filename = f"{filename.replace('_D.tif', '')}.png"
        
        if target_set == "train":
            target_path = os.path.join(folder_C_train, new_filename)
        else:  # val
            target_path = os.path.join(folder_C_val, new_filename)
            
        # 转换并保存图像
        try:
            with Image.open(file_path) as img:
                img.save(target_path, "PNG")
                
                # 如果需要创建测试集，也复制到测试集
                if create_test_folder and target_set == "val":
                    test_path = os.path.join(folder_C_test, new_filename)
                    img.save(test_path, "PNG")
        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")
    
    # 处理标签图像
    print("处理标签图像...")
    for file_path in tqdm(files_E):
        filename = os.path.basename(file_path)
        original_name = extract_original_base_name(filename)
        
        # 跳过不在选择列表中的原始图像
        if original_name not in original_to_target_set:
            continue
            
        target_set = original_to_target_set[original_name]
        new_filename = f"{filename.replace('_E.png', '')}.png"
        
        if target_set == "train":
            target_path = os.path.join(folder_label_train, new_filename)
        else:  # val
            target_path = os.path.join(folder_label_val, new_filename)
            
        # 复制标签图像
        try:
            shutil.copy2(file_path, target_path)
            
            # 如果需要创建测试集，也复制到测试集
            if create_test_folder and target_set == "val":
                test_path = os.path.join(folder_label_test, new_filename)
                shutil.copy2(file_path, test_path)
        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")

    print("处理完成！")
    
    # 计算实际处理的图像总数
    train_A_count = len(os.listdir(folder_A_train))
    val_A_count = len(os.listdir(folder_A_val))
    test_A_count = len(os.listdir(folder_A_test)) if create_test_folder else 0
    
    print(f"训练集大小: {train_A_count} 图像 ({train_A_count/(train_A_count+val_A_count)*100:.1f}%)")
    print(f"验证集大小: {val_A_count} 图像 ({val_A_count/(train_A_count+val_A_count)*100:.1f}%)")
    if create_test_folder:
        print(f"测试集大小: {test_A_count} 图像")
    print(f"输出目录: {output_dir}")


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

        if not a_b_consistent:
            a_minus_b = a_files - b_files
            b_minus_a = b_files - a_files
            if a_minus_b:
                print(f"      A 中独有的文件: {len(a_minus_b)} 个")
            if b_minus_a:
                print(f"      B 中独有的文件: {len(b_minus_a)} 个")

        if not a_c_consistent:
            a_minus_c = a_files - c_files
            c_minus_a = c_files - a_files
            if a_minus_c:
                print(f"      A 中独有的文件: {len(a_minus_c)} 个")
            if c_minus_a:
                print(f"      C 中独有的文件: {len(c_minus_a)} 个")

        if not a_label_consistent:
            a_minus_label = a_files - label_files
            label_minus_a = label_files - a_files
            if a_minus_label:
                print(f"      A 中独有的文件: {len(a_minus_label)} 个")
            if label_minus_a:
                print(f"      label 中独有的文件: {len(label_minus_a)} 个")

    print("\n验证完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理和重组图像数据集")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_DIR, help="输入目录")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument("--val_ratio", type=float, default=DEFAULT_VAL_RATIO, help="验证集比例 (0-1)")
    parser.add_argument("--test", action="store_true", default=DEFAULT_TEST_FOLDER, help="创建测试集文件夹")
    parser.add_argument("--no-test", action="store_false", dest="test", help="不创建测试集文件夹")
    parser.add_argument("--check", action="store_true", default=DEFAULT_CHECK_FORMAT, help="检查图像格式")
    parser.add_argument("--no-check", action="store_false", dest="check", help="不检查图像格式")
    parser.add_argument("--sort", action="store_true", default=DEFAULT_SORT_BY_CHANGE_AREA, help="根据变化区域排序")
    parser.add_argument("--no-sort", action="store_false", dest="sort", help="不根据变化区域排序")
    parser.add_argument("--select", type=float, default=DEFAULT_SELECT_PERCENTAGE, help="选择图像的百分比 (0-100)")
    parser.add_argument("--verify", action="store_true", default=DEFAULT_VERIFY, help="验证数据集结构")

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    if args.verify:
        # 如果只需验证数据集结构，不需要重新处理数据
        verify_dataset_structure(args.output)
    else:
        # 显示运行参数
        print(f"运行参数:")
        print(f"  输入目录: {args.input}")
        print(f"  输出目录: {args.output}")
        print(f"  验证集比例: {args.val_ratio}")
        print(f"  创建测试集: {args.test}")
        print(f"  检查格式: {args.check}")
        print(f"  根据变化区域排序: {args.sort}")
        print(f"  选择图像百分比: {args.select}%")

        # 执行数据处理
        reorganize_processed_images(
            args.input,
            args.output,
            val_ratio=args.val_ratio,
            create_test_folder=args.test,
            check_format=args.check,
            sort_by_change_area=args.sort,
            select_percentage=args.select
        )

        # 处理完后验证数据集结构
        print("\n验证输出数据集结构:")
        verify_dataset_structure(args.output)
