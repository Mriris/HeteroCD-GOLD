import os
import subprocess
import time

def generate_plantuml(puml_files, base_dir='.'):
    """
    使用PlantUML生成图表
    
    Args:
        puml_files: PlantUML文件路径列表(包含子目录)
        base_dir: 基础目录，PUML文件所在父目录
    """
    # 检查是否安装了Java
    try:
        print("检查Java是否安装...")
        subprocess.run(['java', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=10)
        print("Java已安装√")
    except subprocess.TimeoutExpired:
        print("错误: 检查Java安装时超时")
        return None
    except subprocess.CalledProcessError:
        print("错误: 需要安装Java来运行PlantUML")
        return None
    except Exception as e:
        print(f"检查Java时发生未知错误: {e}")
        return None
    
    # 下载PlantUML jar（如果不存在）
    plantuml_jar = os.path.join(base_dir, 'plantuml.jar')
    if not os.path.exists(plantuml_jar):
        try:
            print("下载PlantUML.jar...")
            import urllib.request
            urllib.request.urlretrieve(
                "https://sourceforge.net/projects/plantuml/files/plantuml.jar/download",
                plantuml_jar
            )
            print(f"PlantUML.jar已下载到 {plantuml_jar}")
        except Exception as e:
            print(f"下载PlantUML.jar失败: {e}")
            return None
    else:
        print(f"使用已存在的PlantUML.jar: {plantuml_jar}")
    
    generated_files = []
    
    # 单独处理每个文件，确保一个文件的错误不会影响其他文件的处理
    for puml_file_path in puml_files:
        try:
            # 构造完整的文件路径
            puml_file = os.path.join(base_dir, puml_file_path)
            output_dir = os.path.dirname(puml_file)
            puml_file_name = os.path.basename(puml_file)
            
            print(f"\n处理图表文件: {puml_file}")
            start_time = time.time()
            
            # 检查文件是否存在
            if not os.path.exists(puml_file):
                print(f"错误: 文件 {puml_file} 不存在")
                continue
            
            # 跳过语法检查，直接生成图表
            print(f"直接尝试生成图表...")
            output_file = os.path.splitext(puml_file)[0]
            
            try:
                # 设置超时时间为30秒，避免卡死
                subprocess.run(
                    ['java', '-jar', plantuml_jar, '-tpng', puml_file, '-o', output_dir],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=30  # 设置30秒超时
                )
                print(f"PlantUML命令已执行，耗时: {time.time() - start_time:.2f}秒")
            except subprocess.TimeoutExpired:
                print(f"错误: 生成图表时超时 (>30秒)")
                continue
            
            # 检查是否成功生成图片
            png_file = f"{output_file}.png"
            if os.path.exists(png_file):
                print(f"✓ 图表已成功生成: {png_file}")
                generated_files.append(png_file)
            else:
                print(f"× 生成图表失败: {png_file}")
            
        except subprocess.CalledProcessError as e:
            print(f"处理 {puml_file_path} 时发生错误: {e}")
            if hasattr(e, 'stdout') and e.stdout:
                try:
                    print(f"输出: {e.stdout.decode('utf-8', errors='replace')}")
                except Exception:
                    print("无法解码标准输出")
            if hasattr(e, 'stderr') and e.stderr:
                try:
                    print(f"错误: {e.stderr.decode('utf-8', errors='replace')}")
                except Exception:
                    print("无法解码标准错误输出")
        except Exception as e:
            print(f"处理 {puml_file_path} 时发生未知错误: {type(e).__name__}: {e}")
    
    return generated_files if generated_files else None

def find_puml_files(base_dir, subdirs=None):
    """查找指定子目录中的所有PUML文件"""
    all_puml_files = []
    
    if subdirs:
        # 只搜索指定的子目录
        for subdir in subdirs:
            subdir_path = os.path.join(base_dir, subdir)
            if os.path.isdir(subdir_path):
                for file in os.listdir(subdir_path):
                    if file.endswith('.puml'):
                        relative_path = os.path.join(subdir, file)
                        all_puml_files.append(relative_path)
    else:
        # 搜索所有子目录
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.puml'):
                    rel_path = os.path.relpath(os.path.join(root, file), base_dir)
                    all_puml_files.append(rel_path)
    
    return all_puml_files

if __name__ == "__main__":
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 定义要处理的子目录
    subdirs = ["差异图注意力迁移", "双向通道注意力", "在线蒸馏"]
    
    # 查找所有PUML文件
    all_puml_files = find_puml_files(script_dir, subdirs)
    
    # 特定的PUML文件列表（如果要指定生成哪些文件）
    # 注意:路径需要使用相对于script_dir的路径
    selected_puml_files = [
        # 差异图注意力迁移文件
        "差异图注意力迁移/差异图注意力迁移流程图.puml",
        "差异图注意力迁移/差异图注意力迁移流程图-简化版.puml",
        "差异图注意力迁移/差异图注意力迁移-多度量差异图生成.puml",
        "差异图注意力迁移/差异图注意力迁移-双维度注意力计算.puml",
        "差异图注意力迁移/差异图注意力迁移-注意力迁移损失计算.puml",
        
        # 双向通道注意力文件
        "双向通道注意力/双向通道注意力流程图.puml",
        "双向通道注意力/双向通道注意力流程图-简化版.puml",
        "双向通道注意力/双向通道注意力-计算通道注意力权重.puml",
        "双向通道注意力/双向通道注意力-共享注意力计算.puml",
        "双向通道注意力/双向通道注意力-特征增强与融合.puml",
        
        # 在线蒸馏文件
        "在线蒸馏/在线蒸馏流程图.puml",
        "在线蒸馏/在线蒸馏流程图-简化版.puml",
        "在线蒸馏/在线蒸馏-特征提取阶段.puml",
        "在线蒸馏/在线蒸馏-多层次知识蒸馏.puml",
        "在线蒸馏/在线蒸馏-总损失函数计算.puml"
    ]
    
    # 取消下面注释可以生成所有找到的PUML文件
    # puml_files = all_puml_files
    
    # 使用特定的PUML文件列表
    puml_files = selected_puml_files
    
    print(f"脚本所在目录: {script_dir}")
    print(f"找到 {len(all_puml_files)} 个PUML文件")
    print(f"将要生成 {len(puml_files)} 个流程图")
    print("开始生成流程图...")
    
    output_paths = generate_plantuml(puml_files, base_dir=script_dir)
    
    if output_paths and len(output_paths) > 0:
        print("\n成功生成的流程图:")
        for i, path in enumerate(output_paths, 1):
            print(f"{i}. {path}")
    else:
        print("\n没有成功生成任何流程图")
    
    print("程序执行完毕") 