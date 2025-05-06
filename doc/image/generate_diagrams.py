import os
import subprocess
import time

def generate_plantuml(puml_files, output_dir='.'):
    """
    使用PlantUML生成图表
    
    Args:
        puml_files: PlantUML文件路径列表
        output_dir: 输出目录
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
    plantuml_jar = os.path.join(output_dir, 'plantuml.jar')
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
    for puml_file in puml_files:
        try:
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
            print(f"处理 {puml_file} 时发生错误: {e}")
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
            print(f"处理 {puml_file} 时发生未知错误: {type(e).__name__}: {e}")
    
    return generated_files if generated_files else None

if __name__ == "__main__":
    puml_files = [
        "双向通道注意力流程图.puml",
        "GOLD模型整体框架图.puml",
        "在线蒸馏机制流程图.puml",
        "差异图注意力迁移流程图.puml",
        "自适应动态权重流程图.puml"
    ]
    
    print("开始生成流程图...")
    output_paths = generate_plantuml(puml_files)
    
    if output_paths and len(output_paths) > 0:
        print("\n成功生成的流程图:")
        for i, path in enumerate(output_paths, 1):
            print(f"{i}. {os.path.abspath(path)}")
    else:
        print("\n没有成功生成任何流程图")
    
    print("程序执行完毕") 