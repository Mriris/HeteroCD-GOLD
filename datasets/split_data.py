import os
import argparse
from PIL import Image
from tqdm import tqdm
import glob
import random
import numpy as np
from pathlib import Path
import shutil

# 默认参数设置（放在代码最上方以便于修改）
DEFAULT_INPUT_DIR = "/data/jingwei/yantingxuan/Datasets/CityCN/Enhanced"
DEFAULT_OUTPUT_DIR = "/data/jingwei/yantingxuan/Datasets/CityCN/Test"
DEFAULT_VERIFY = False  # 是否默认验证数据集完整性
DEFAULT_VAL_RATIO = 0.2  # 验证集占总数据的比例
DEFAULT_TEST_FOLDER = True  # 是否创建测试集文件夹（内容与验证集相同）
DEFAULT_CHECK_FORMAT = False  # 是否检查图像格式
DEFAULT_SORT_BY_CHANGE_AREA = True  # 是否根据变化区域大小进行排序选择
DEFAULT_SELECT_PERCENTAGE = 1.0  # 选择的图像百分比

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

def reorganize_processed_images(input_dir, output_dir, val_ratio=0.2, create_test_folder=True, check_format=True,
                                sort_by_change_area=False, select_percentage=100.0):
    """
    重新组织处理后的图像文件:
    1. _A.tif 文件移到 train/A, val/A 和 test/A 文件夹，并转换为 png
    2. _B.tif 文件移到 train/B, val/B 和 test/B 文件夹，并转换为 png
    3. _D.tif 文件移到 train/C, val/C 和 test/C 文件夹，并转换为 png
    4. _E.png 文件移到 train/label, val/label 和 test/label 文件夹
    5. 所有文件名去掉末尾的 _X 标识
    6. 可根据变化区域大小排序并选择处理指定百分比的图像
    7. 测试集与验证集内容相同

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

    # 获取图像名称列表（去除后缀）
    base_names = [os.path.basename(file).replace("_A.tif", "") for file in files_A]

    # 如果需要根据变化区域大小排序
    if sort_by_change_area:
        print("根据变化区域大小排序图像...")
        area_data = []
        for base_name in tqdm(base_names, desc="计算变化区域"):
            label_path = os.path.join(input_dir, f"{base_name}_E.png")
            if os.path.exists(label_path):
                area = calculate_change_area(label_path)
                area_data.append((base_name, area))

        # 根据变化区域大小降序排序
        area_data.sort(key=lambda x: x[1], reverse=True)

        # 选择前 select_percentage% 的图像
        select_count = int(len(area_data) * (select_percentage / 100.0))
        selected_base_names = [data[0] for data in area_data[:select_count]]

        print(f"根据变化区域大小选择了 {len(selected_base_names)} 个图像 ({select_percentage}%)")

        # 统计选择的图像的变化区域大小分布
        if selected_base_names:
            areas = [data[1] for data in area_data[:select_count]]
            print(f"选择的图像变化区域大小统计:")
            print(f"  最小值: {min(areas)}")
            print(f"  最大值: {max(areas)}")
            print(f"  平均值: {sum(areas) / len(areas):.2f}")
            print(f"  中位数: {sorted(areas)[len(areas) // 2]}")
    else:
        selected_base_names = base_names
        print(f"将处理所有 {len(selected_base_names)} 个图像")

    # 随机打乱图像顺序
    random.shuffle(selected_base_names)

    # 确定验证集大小
    val_size = int(len(selected_base_names) * val_ratio)

    # 划分训练集和验证集
    val_names = selected_base_names[:val_size]
    train_names = selected_base_names[val_size:]

    print(f"分配 {len(train_names)} 个图像到训练集，{len(val_names)} 个图像到验证集")

    # 处理训练集图像
    for base_name in tqdm(train_names, desc="处理训练集图像"):
        file_A = os.path.join(input_dir, f"{base_name}_A.tif")
        file_B = os.path.join(input_dir, f"{base_name}_B.tif")
        file_C = os.path.join(input_dir, f"{base_name}_D.tif")  # 修改为 _D.tif
        file_E = os.path.join(input_dir, f"{base_name}_E.png")

        # 检查文件是否存在
        if not all(os.path.exists(file) for file in [file_A, file_B, file_C, file_E]):
            print(f"警告: {base_name} 缺少部分文件，跳过")
            continue

        # 处理 A 类图像
        target_A = os.path.join(folder_A_train, f"{base_name}.png")
        try:
            with Image.open(file_A) as img:
                img.save(target_A, "PNG")
        except Exception as e:
            print(f"处理 {base_name}_A.tif 时出错: {e}")

        # 处理 B 类图像
        target_B = os.path.join(folder_B_train, f"{base_name}.png")
        try:
            with Image.open(file_B) as img:
                img.save(target_B, "PNG")
        except Exception as e:
            print(f"处理 {base_name}_B.tif 时出错: {e}")

        # 处理 C 类图像
        target_C = os.path.join(folder_C_train, f"{base_name}.png")
        try:
            with Image.open(file_C) as img:
                img.save(target_C, "PNG")
        except Exception as e:
            print(f"处理 {base_name}_D.tif 时出错: {e}")

        # 处理标签图像，直接复制
        target_E = os.path.join(folder_label_train, f"{base_name}.png")
        try:
            shutil.copy2(file_E, target_E)
        except Exception as e:
            print(f"处理 {base_name}_E.png 时出错: {e}")

    # 处理验证集图像
    for base_name in tqdm(val_names, desc="处理验证集图像"):
        file_A = os.path.join(input_dir, f"{base_name}_A.tif")
        file_B = os.path.join(input_dir, f"{base_name}_B.tif")
        file_C = os.path.join(input_dir, f"{base_name}_D.tif")  # 修改为 _D.tif
        file_E = os.path.join(input_dir, f"{base_name}_E.png")

        # 检查文件是否存在
        if not all(os.path.exists(file) for file in [file_A, file_B, file_C, file_E]):
            print(f"警告: {base_name} 缺少部分文件，跳过")
            continue

        # 处理 A 类图像
        target_A = os.path.join(folder_A_val, f"{base_name}.png")
        try:
            with Image.open(file_A) as img:
                img.save(target_A, "PNG")
        except Exception as e:
            print(f"处理 {base_name}_A.tif 时出错: {e}")

        # 处理 B 类图像
        target_B = os.path.join(folder_B_val, f"{base_name}.png")
        try:
            with Image.open(file_B) as img:
                img.save(target_B, "PNG")
        except Exception as e:
            print(f"处理 {base_name}_B.tif 时出错: {e}")

        # 处理 C 类图像
        target_C = os.path.join(folder_C_val, f"{base_name}.png")
        try:
            with Image.open(file_C) as img:
                img.save(target_C, "PNG")
        except Exception as e:
            print(f"处理 {base_name}_D.tif 时出错: {e}")

        # 处理标签图像，直接复制
        target_E = os.path.join(folder_label_val, f"{base_name}.png")
        try:
            shutil.copy2(file_E, target_E)
        except Exception as e:
            print(f"处理 {base_name}_E.png 时出错: {e}")

    # 处理测试集图像（如果需要）
    if create_test_folder:
        for base_name in tqdm(val_names, desc="处理测试集图像"):
            file_A = os.path.join(input_dir, f"{base_name}_A.tif")
            file_B = os.path.join(input_dir, f"{base_name}_B.tif")
            file_C = os.path.join(input_dir, f"{base_name}_D.tif")  # 修改为 _D.tif
            file_E = os.path.join(input_dir, f"{base_name}_E.png")

            # 检查文件是否存在
            if not all(os.path.exists(file) for file in [file_A, file_B, file_C, file_E]):
                print(f"警告: {base_name} 缺少部分文件，跳过")
                continue

            # 处理 A 类图像
            target_A = os.path.join(folder_A_test, f"{base_name}.png")
            try:
                with Image.open(file_A) as img:
                    img.save(target_A, "PNG")
            except Exception as e:
                print(f"处理 {base_name}_A.tif 时出错: {e}")

            # 处理 B 类图像
            target_B = os.path.join(folder_B_test, f"{base_name}.png")
            try:
                with Image.open(file_B) as img:
                    img.save(target_B, "PNG")
            except Exception as e:
                print(f"处理 {base_name}_B.tif 时出错: {e}")

            # 处理 C 类图像
            target_C = os.path.join(folder_C_test, f"{base_name}.png")
            try:
                with Image.open(file_C) as img:
                    img.save(target_C, "PNG")
            except Exception as e:
                print(f"处理 {base_name}_D.tif 时出错: {e}")

            # 处理标签图像，直接复制
            target_E = os.path.join(folder_label_test, f"{base_name}.png")
            try:
                shutil.copy2(file_E, target_E)
            except Exception as e:
                print(f"处理 {base_name}_E.png 时出错: {e}")

    print("处理完成！")
    print(f"训练集大小: {len(train_names)}")
    print(f"验证集大小: {len(val_names)}")
    if create_test_folder:
        print(f"测试集大小: {len(val_names)}")
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
