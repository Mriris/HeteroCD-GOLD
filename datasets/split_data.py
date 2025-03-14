import os
import argparse
from PIL import Image
from tqdm import tqdm
import glob
import random
import numpy as np
from pathlib import Path

# 默认参数设置（放在代码最上方以便于修改）
DEFAULT_INPUT_DIR = "/data/jingwei/yantingxuan/Datasets/CityCN/Enhanced"
DEFAULT_OUTPUT_DIR = "/data/jingwei/yantingxuan/Datasets/CityCN/Split"
DEFAULT_VERIFY = False  # 是否默认验证数据集完整性
DEFAULT_VAL_RATIO = 0.2  # 验证集占总数据的比例
DEFAULT_CHECK_FORMAT = False  # 是否检查图像格式
DEFAULT_SORT_BY_CHANGE_AREA = False  # 是否根据变化区域大小进行排序选择
DEFAULT_SELECT_PERCENTAGE = 30.0  # 默认选择100%的图像（即不筛选）

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

def reorganize_processed_images(input_dir, output_dir, val_ratio=0.2, check_format=True,
                                sort_by_change_area=False, select_percentage=100.0):
    """
    重新组织处理后的图像文件:
    1. _A.tif 文件移到 train/A 和 val/A 文件夹，并转换为 png
    2. _B.tif 文件移到 train/B 和 val/B 文件夹，并转换为 png
    3. _E.png 文件移到 train/label 和 val/label 文件夹
    4. 所有文件名去掉末尾的 _X 标识
    5. 可根据变化区域大小排序并选择处理指定百分比的图像

    参数:
        input_dir: 处理后图像所在的输入目录
        output_dir: 重组后的输出目录基础路径
        val_ratio: 验证集占总数据的比例
        check_format: 是否检查图像格式
        sort_by_change_area: 是否根据变化区域大小进行排序选择
        select_percentage: 选择的图像百分比（0-100）
    """
    # 创建输出目录结构
    train_folder = os.path.join(output_dir, "train")
    val_folder = os.path.join(output_dir, "val")

    folder_A_train = os.path.join(train_folder, "A")
    folder_B_train = os.path.join(train_folder, "B")
    folder_label_train = os.path.join(train_folder, "label")

    folder_A_val = os.path.join(val_folder, "A")
    folder_B_val = os.path.join(val_folder, "B")
    folder_label_val = os.path.join(val_folder, "label")

    for folder in [folder_A_train, folder_B_train, folder_label_train,
                   folder_A_val, folder_B_val, folder_label_val]:
        os.makedirs(folder, exist_ok=True)

    # 查找所有相关文件
    files_A = glob.glob(os.path.join(input_dir, "*_A.tif"))
    files_B = glob.glob(os.path.join(input_dir, "*_B.tif"))
    files_E = glob.glob(os.path.join(input_dir, "*_E.png"))

    print(f"找到 {len(files_A)} 个 A 类图像")
    print(f"找到 {len(files_B)} 个 B 类图像")
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
                            print(f"  模式: {img.mode}, 尺寸: {img.size}, 唯一值: {unique_values[:20]}")
                            if len(unique_values) > 20:
                                print(f"  ... 共计 {len(unique_values)} 个唯一值")
                            print(f"  值范围: [{img_array.min()}, {img_array.max()}]")

                            if len(unique_values) < 256:
                                value_counts = {}
                                for val in unique_values:
                                    count = np.sum(img_array == val)
                                    value_counts[int(val)] = count

                                print("  值分布:")
                                for val, count in sorted(value_counts.items())[:10]:
                                    percentage = (count / img_array.size) * 100
                                    print(f"    - 值 {val}: {count} 像素 ({percentage:.2f}%)")
                                if len(value_counts) > 10:
                                    print(f"    ... 以及其他 {len(value_counts) - 10} 个值")
                    except Exception as e:
                        print(f"  分析时出错: {e}")
                    print()

            proceed = input("图像格式检查发现问题，是否继续处理? (y/n): ")
            if proceed.lower() != 'y':
                print("用户取消操作")
                return None, None

            if label_format_issues:
                fix_labels = input("是否尝试修复标签图像为严格二值图? (y/n): ")
                if fix_labels.lower() == 'y':
                    threshold = input("请输入二值化阈值(0-255，默认128): ")
                    try:
                        threshold = int(threshold) if threshold.strip() else 128
                        threshold = max(0, min(255, threshold))
                        print(f"将使用阈值 {threshold} 进行二值化处理")
                    except ValueError:
                        threshold = 128
                        print(f"输入无效，将使用默认阈值 {threshold}")
                else:
                    print("将保持标签图像原始值")

    # 获取所有基础名称（基于标签图像）
    base_names = []
    for file_path in files_E:
        filename = os.path.basename(file_path)
        base_name = filename[:-6] if filename.endswith("_E.png") else filename
        base_names.append(base_name)

    # 如果需要根据变化区域大小排序
    if sort_by_change_area:
        print("正在计算变化区域大小并排序...")
        change_areas = []
        for base_name in tqdm(base_names, desc="计算变化区域"):
            label_path = os.path.join(input_dir, f"{base_name}_E.png")
            area = calculate_change_area(label_path)
            change_areas.append((base_name, area))

        # 按变化区域大小从大到小排序
        change_areas.sort(key=lambda x: x[1], reverse=True)

        # 选择前 select_percentage% 的图像
        select_count = max(1, int(len(change_areas) * (select_percentage / 100.0)))  # 至少选择1张
        selected_base_names = [item[0] for item in change_areas[:select_count]]
        print(f"选择变化区域最大的 {select_percentage}% 图像，共 {len(selected_base_names)} 个")
    else:
        selected_base_names = base_names
        print("不进行排序选择，处理所有图像")

    # 随机分配训练集和验证集
    random.seed(42)  # 设置随机种子，确保结果可重现
    random.shuffle(selected_base_names)

    val_size = int(len(selected_base_names) * val_ratio)
    val_names = set(selected_base_names[:val_size])
    train_names = set(selected_base_names[val_size:])

    print(f"分配到训练集: {len(train_names)} 个样本")
    print(f"分配到验证集: {len(val_names)} 个样本")

    # 处理每个文件
    processed_count = 0

    # 处理 A 类图像
    for file_path in tqdm(files_A, desc="处理 A 类图像"):
        filename = os.path.basename(file_path)
        base_name = filename[:-6] if filename.endswith("_A.tif") else filename

        if base_name not in selected_base_names:
            continue

        if base_name in train_names:
            target_dir = folder_A_train
        elif base_name in val_names:
            target_dir = folder_A_val
        else:
            continue

        # 新文件路径
        new_file_path = os.path.join(target_dir, f"{base_name}.png")

        # 转换格式并保存
        try:
            img = Image.open(file_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(new_file_path, "PNG")
            processed_count += 1
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")

    # 处理 B 类图像
    for file_path in tqdm(files_B, desc="处理 B 类图像"):
        filename = os.path.basename(file_path)
        base_name = filename[:-6] if filename.endswith("_B.tif") else filename

        if base_name not in selected_base_names:
            continue

        if base_name in train_names:
            target_dir = folder_B_train
        elif base_name in val_names:
            target_dir = folder_B_val
        else:
            continue

        # 新文件路径
        new_file_path = os.path.join(target_dir, f"{base_name}.png")

        # 转换格式并保存
        try:
            img = Image.open(file_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(new_file_path, "PNG")
            processed_count += 1
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")

    # 处理标签图像
    for file_path in tqdm(files_E, desc="处理标签图像"):
        filename = os.path.basename(file_path)
        base_name = filename[:-6] if filename.endswith("_E.png") else filename

        if base_name not in selected_base_names:
            continue

        if base_name in train_names:
            target_dir = folder_label_train
        elif base_name in val_names:
            target_dir = folder_label_val
        else:
            continue

        # 新文件路径
        new_file_path = os.path.join(target_dir, f"{base_name}.png")

        # 复制文件并确保为二值图
        try:
            img = Image.open(file_path)
            if img.mode != 'L':
                img = img.convert('L')

            img_array = np.array(img)
            if check_format and 'threshold' in locals() and 'fix_labels' in locals() and fix_labels.lower() == 'y':
                img_array = np.where(img_array > threshold, 255, 0).astype(np.uint8)
            else:
                threshold = 128
                img_array = np.where(img_array > threshold, 255, 0).astype(np.uint8)

            binary_img = Image.fromarray(img_array)
            binary_img.save(new_file_path, "PNG")
            processed_count += 1
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")

    print(f"处理完成！文件已组织到以下目录：")
    print(f"训练集 A 类图像: {folder_A_train}")
    print(f"训练集 B 类图像: {folder_B_train}")
    print(f"训练集标签图像: {folder_label_train}")
    print(f"验证集 A 类图像: {folder_A_val}")
    print(f"验证集 B 类图像: {folder_B_val}")
    print(f"验证集标签图像: {folder_label_val}")

    # 检查文件数量是否一致
    train_count_A = len(glob.glob(os.path.join(folder_A_train, "*.png")))
    train_count_B = len(glob.glob(os.path.join(folder_B_train, "*.png")))
    train_count_E = len(glob.glob(os.path.join(folder_label_train, "*.png")))

    val_count_A = len(glob.glob(os.path.join(folder_A_val, "*.png")))
    val_count_B = len(glob.glob(os.path.join(folder_B_val, "*.png")))
    val_count_E = len(glob.glob(os.path.join(folder_label_val, "*.png")))

    print(f"最终文件数量统计:")
    print(f"训练集 A 类图像: {train_count_A}")
    print(f"训练集 B 类图像: {train_count_B}")
    print(f"训练集标签图像: {train_count_E}")
    print(f"验证集 A 类图像: {val_count_A}")
    print(f"验证集 B 类图像: {val_count_B}")
    print(f"验证集标签图像: {val_count_E}")

    train_ok = (train_count_A == train_count_B == train_count_E)
    val_ok = (val_count_A == val_count_B == val_count_E)

    if train_ok and val_ok:
        print("所有类别的图像数量一致，整理成功！")
    else:
        print("警告: 各类别图像数量不一致，请检查是否有文件丢失或处理失败。")

    return train_folder, val_folder

def verify_dataset_integrity(train_folder, val_folder, check_format=True):
    """
    验证数据集在训练集和验证集中的完整性

    参数:
        train_folder: 训练集目录
        val_folder: 验证集目录
        check_format: 是否检查图像格式
    """
    if not train_folder or not val_folder:
        print("未提供有效的数据集目录，跳过验证")
        return False

    folder_A_train = os.path.join(train_folder, "A")
    folder_B_train = os.path.join(train_folder, "B")
    folder_label_train = os.path.join(train_folder, "label")

    folder_A_val = os.path.join(val_folder, "A")
    folder_B_val = os.path.join(val_folder, "B")
    folder_label_val = os.path.join(val_folder, "label")

    # 验证训练集
    print("验证训练集完整性...")
    train_files_A = set([os.path.basename(f) for f in glob.glob(os.path.join(folder_A_train, "*.png"))])
    train_files_B = set([os.path.basename(f) for f in glob.glob(os.path.join(folder_B_train, "*.png"))])
    train_files_label = set([os.path.basename(f) for f in glob.glob(os.path.join(folder_label_train, "*.png"))])

    all_train_files = train_files_A.union(train_files_B).union(train_files_label)

    missing_in_A_train = all_train_files - train_files_A
    missing_in_B_train = all_train_files - train_files_B
    missing_in_label_train = all_train_files - train_files_label

    train_integrity_ok = True

    if missing_in_A_train:
        print(f"在训练集 A 文件夹中缺少 {len(missing_in_A_train)} 个文件")
        train_integrity_ok = False

    if missing_in_B_train:
        print(f"在训练集 B 文件夹中缺少 {len(missing_in_B_train)} 个文件")
        train_integrity_ok = False

    if missing_in_label_train:
        print(f"在训练集 label 文件夹中缺少 {len(missing_in_label_train)} 个文件")
        train_integrity_ok = False

    if train_integrity_ok:
        print(f"训练集完整性检查通过！所有 {len(all_train_files)} 个样本在三个文件夹中都有对应文件。")
    else:
        print("训练集完整性检查失败，请检查上述缺失文件。")

    # 验证验证集
    print("\n验证验证集完整性...")
    val_files_A = set([os.path.basename(f) for f in glob.glob(os.path.join(folder_A_val, "*.png"))])
    val_files_B = set([os.path.basename(f) for f in glob.glob(os.path.join(folder_B_val, "*.png"))])
    val_files_label = set([os.path.basename(f) for f in glob.glob(os.path.join(folder_label_val, "*.png"))])

    all_val_files = val_files_A.union(val_files_B).union(val_files_label)

    missing_in_A_val = all_val_files - val_files_A
    missing_in_B_val = all_val_files - val_files_B
    missing_in_label_val = all_val_files - val_files_label

    val_integrity_ok = True

    if missing_in_A_val:
        print(f"在验证集 A 文件夹中缺少 {len(missing_in_A_val)} 个文件")
        val_integrity_ok = False

    if missing_in_B_val:
        print(f"在验证集 B 文件夹中缺少 {len(missing_in_B_val)} 个文件")
        val_integrity_ok = False

    if missing_in_label_val:
        print(f"在验证集 label 文件夹中缺少 {len(missing_in_label_val)} 个文件")
        val_integrity_ok = False

    if val_integrity_ok:
        print(f"验证集完整性检查通过！所有 {len(all_val_files)} 个样本在三个文件夹中都有对应文件。")
    else:
        print("验证集完整性检查失败，请检查上述缺失文件。")

    # 检查训练集和验证集是否有重叠
    overlap = train_files_A.intersection(val_files_A)
    if overlap:
        print(f"警告: 训练集和验证集有 {len(overlap)} 个重叠样本！")
        for f in sorted(overlap)[:5]:
            print(f"  - {f}")
        if len(overlap) > 5:
            print(f"  ... 以及其他 {len(overlap) - 5} 个文件")
        return False

    # 检查图像格式
    if check_format:
        print("\n检查处理后的图像格式...")
        format_errors = []

        sample_size = min(5, len(train_files_A))

        if train_files_A:
            train_A_samples = random.sample(glob.glob(os.path.join(folder_A_train, "*.png")), sample_size)
            for file_path in train_A_samples:
                passed, error_msg, details = check_image_format(file_path, required_bands=3)
                if not passed:
                    format_errors.append(f"训练集A类图像 {os.path.basename(file_path)}: {error_msg} | {details}")

        if train_files_B:
            train_B_samples = random.sample(glob.glob(os.path.join(folder_B_train, "*.png")), sample_size)
            for file_path in train_B_samples:
                passed, error_msg, details = check_image_format(file_path, required_bands=3)
                if not passed:
                    format_errors.append(f"训练集B类图像 {os.path.basename(file_path)}: {error_msg} | {details}")

        if train_files_label:
            train_label_samples = random.sample(glob.glob(os.path.join(folder_label_train, "*.png")), sample_size)
            for file_path in train_label_samples:
                passed, error_msg, details = check_image_format(file_path, required_mode='L')
                if not passed:
                    format_errors.append(f"训练集标签图像 {os.path.basename(file_path)}: {error_msg} | {details}")

        if len(val_files_A) >= sample_size:
            val_A_samples = random.sample(glob.glob(os.path.join(folder_A_val, "*.png")), sample_size)
            for file_path in val_A_samples:
                passed, error_msg, details = check_image_format(file_path, required_bands=3)
                if not passed:
                    format_errors.append(f"验证集A类图像 {os.path.basename(file_path)}: {error_msg} | {details}")

            val_B_samples = random.sample(glob.glob(os.path.join(folder_B_val, "*.png")), sample_size)
            for file_path in val_B_samples:
                passed, error_msg, details = check_image_format(file_path, required_bands=3)
                if not passed:
                    format_errors.append(f"验证集B类图像 {os.path.basename(file_path)}: {error_msg} | {details}")

            val_label_samples = random.sample(glob.glob(os.path.join(folder_label_val, "*.png")), sample_size)
            for file_path in val_label_samples:
                passed, error_msg, details = check_image_format(file_path, required_mode='L')
                if not passed:
                    format_errors.append(f"验证集标签图像 {os.path.basename(file_path)}: {error_msg} | {details}")

        if format_errors:
            print("发现图像格式问题:")
            for error in format_errors:
                print(f"  - {error}")
            return False

    return train_integrity_ok and val_integrity_ok

def main():
    """主函数，处理命令行参数并执行相应操作"""
    parser = argparse.ArgumentParser(description="图像数据集组织工具")
    parser.add_argument('--input_dir', type=str, default=DEFAULT_INPUT_DIR,
                        help=f'待处理图像所在目录 (默认: {DEFAULT_INPUT_DIR})')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'输出目录基础路径 (默认: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--val_ratio', type=float, default=DEFAULT_VAL_RATIO,
                        help=f'验证集占比 (默认: {DEFAULT_VAL_RATIO})')
    parser.add_argument('--verify', action='store_true', default=DEFAULT_VERIFY,
                        help='是否验证数据集完整性')
    parser.add_argument('--check_format', action='store_true', default=DEFAULT_CHECK_FORMAT,
                        help='是否检查图像格式')
    parser.add_argument('--sort_by_change_area', action='store_true', default=DEFAULT_SORT_BY_CHANGE_AREA,
                        help='是否根据变化区域大小进行排序选择')
    parser.add_argument('--select_percentage', type=float, default=DEFAULT_SELECT_PERCENTAGE,
                        help='选择的图像百分比 (0-100)')

    args = parser.parse_args()

    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"验证集占比: {args.val_ratio}")
    print(f"是否验证数据集: {args.verify}")
    print(f"是否检查图像格式: {args.check_format}")
    print(f"是否根据变化区域排序: {args.sort_by_change_area}")
    print(f"选择的图像百分比: {args.select_percentage}%")

    # 检查输入目录是否存在
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录 {args.input_dir} 不存在!")
        return

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 执行数据集重组
    train_folder, val_folder = reorganize_processed_images(
        args.input_dir, args.output_dir,
        val_ratio=args.val_ratio,
        check_format=args.check_format,
        sort_by_change_area=args.sort_by_change_area,
        select_percentage=args.select_percentage
    )

    # 如果需要，验证数据集完整性
    if args.verify:
        verify_dataset_integrity(train_folder, val_folder, check_format=args.check_format)

if __name__ == "__main__":
    main()
