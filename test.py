import os
import time
import datetime
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
import subprocess
import threading
import re
import pandas as pd
from collections import defaultdict

from models.GOLD import TripleHeteCD
from options.test_options import TestOptions
from datasets import dataset
from utils.util import get_confuse_matrix, cm2score

# 添加彩色比对图生成函数
def generate_comparison_image(pred, label):
    """生成彩色比对图，用于可视化多检和漏检情况
    
    Args:
        pred (numpy.ndarray): 预测结果, 形状为 (H, W), 值为0或1的二值图
        label (numpy.ndarray): 真值标签, 形状为 (H, W), 值为0或1的二值图
        
    Returns:
        numpy.ndarray: 彩色比对图, 形状为 (H, W, 3), RGB格式
            - 白色 (255,255,255): 真正例 (真实变化且预测为变化)
            - 黑色 (0,0,0): 真负例 (真实无变化且预测为无变化)
            - 红色 (255,0,0): 漏检 (真实变化但预测为无变化)
            - 绿色 (0,255,0): 多检 (真实无变化但预测为变化)
    """
    # 确保输入是二值图
    pred = pred.astype(np.bool_)
    label = label.astype(np.bool_)
    
    # 创建空白RGB图像
    h, w = pred.shape
    comparison = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 真正例 - 白色 (255,255,255)
    comparison[np.logical_and(label, pred)] = [255, 255, 255]
    
    # 真负例 - 黑色 (0,0,0) - 默认已是黑色
    
    # 漏检 (假负例) - 红色 (255,0,0) - 真实是变化但预测为无变化
    comparison[np.logical_and(label, np.logical_not(pred))] = [255, 0, 0]
    
    # 多检 (假正例) - 绿色 (0,255,0) - 真实是无变化但预测为变化
    comparison[np.logical_and(np.logical_not(label), pred)] = [0, 255, 0]
    
    return comparison

# GPU监控线程类
class GPUMonitor(threading.Thread):
    def __init__(self, gpu_id, interval=0.1):
        """初始化GPU监控线程
        
        Args:
            gpu_id (int): 要监控的GPU ID
            interval (float): 采样间隔，单位为秒
        """
        threading.Thread.__init__(self)
        self.gpu_id = gpu_id
        self.interval = interval
        self.stopped = False
        self.stats = defaultdict(list)  # 用于存储统计数据
        
    def run(self):
        while not self.stopped:
            try:
                # 使用nvidia-smi获取GPU状态
                cmd = f"nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader,nounits --id={self.gpu_id}"
                output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
                
                # 解析输出结果
                values = output.split(',')
                if len(values) >= 6:
                    # 记录数据
                    self.stats['memory_used'].append(float(values[1].strip()))
                    self.stats['memory_total'].append(float(values[2].strip()))
                    self.stats['utilization'].append(float(values[3].strip()))
                    self.stats['power'].append(float(values[4].strip()))
                    self.stats['temperature'].append(float(values[5].strip()))
                
                # 也记录PyTorch报告的内存使用情况
                if torch.cuda.is_available():
                    self.stats['torch_allocated'].append(torch.cuda.memory_allocated(self.gpu_id) / (1024**2))  # MB
                    self.stats['torch_reserved'].append(torch.cuda.memory_reserved(self.gpu_id) / (1024**2))  # MB
                    
            except Exception as e:
                print(f"GPU监控出错: {e}")
                
            time.sleep(self.interval)
            
    def stop(self):
        """停止监控线程"""
        self.stopped = True
        
    def get_stats(self):
        """获取统计结果
        
        Returns:
            dict: 包含各项统计数据的字典
        """
        result = {}
        
        # 计算各项统计指标
        for key, values in self.stats.items():
            if values:
                result[f"{key}_avg"] = np.mean(values)
                result[f"{key}_max"] = np.max(values)
                result[f"{key}_min"] = np.min(values)
                
        # 计算内存使用率百分比
        if 'memory_used' in self.stats and 'memory_total' in self.stats and self.stats['memory_total']:
            memory_used_percent = [u/t*100 for u, t in zip(self.stats['memory_used'], self.stats['memory_total'])]
            result['memory_percent_avg'] = np.mean(memory_used_percent)
            result['memory_percent_max'] = np.max(memory_used_percent)
            result['memory_percent_min'] = np.min(memory_used_percent)
            
        return result

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

def format_time(seconds):
    """将秒数格式化为更精确的时间字符串，保留毫秒
    
    Args:
        seconds (float): 秒数
        
    Returns:
        str: 格式化的时间字符串，形如 "小时:分钟:秒.毫秒"
    """
    # 分离整数部分和小数部分
    int_seconds = int(seconds)
    milliseconds = int((seconds - int_seconds) * 1000)
    
    # 创建timedelta对象处理小时、分钟和秒
    delta = datetime.timedelta(seconds=int_seconds)
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # 格式化为字符串，带毫秒
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

def test_model(opt):
    # 创建结果目录
    result_dir = os.path.join(opt.results_dir, opt.name)
    pred_dir = os.path.join(result_dir, 'predictions')
    os.makedirs(pred_dir, exist_ok=True)
    
    # 如果启用比对图，创建比对图目录
    if opt.save_comparison:
        comp_dir = os.path.join(result_dir, opt.comparison_dir)
        os.makedirs(comp_dir, exist_ok=True)
        print(f"将生成彩色比对图并保存至: {comp_dir}")
    
    # 设置随机种子
    if hasattr(opt, 'seed'):
        setup_seed(opt.seed)
    
    print(f"{'='*20} 变化检测测试开始 {'='*20}")
    print(f"数据集: {opt.dataroot}")
    print(f"结果保存至: {result_dir}")
    print(f"GPU IDs: {opt.gpu_ids}")
    
    # 加载数据集
    test_set = dataset.Data(opt.phase, root=opt.dataroot, opt=opt, load_t2_opt=opt.load_t2_opt)
    test_loader = DataLoader(test_set, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)
    
    # 创建模型 - 测试时禁用训练配置
    opt.isTrain = False  # 确保处于测试模式
    model = TripleHeteCD(opt, is_train=False)
    
    # 直接加载指定模型权重，跳过setup自动加载
    if opt.model_path is not None:
        model_path = opt.model_path
        print(f"加载指定模型: {model_path}")
        try:
            model.load_weights(model_path)
            print(f"模型成功加载，继续测试...")
        except Exception as e:
            print(f"加载模型出错: {e}")
            return None
    else:
        print("错误: 未指定模型路径，请使用--model_path参数指定模型文件")
        return None
    
    # 确保模型处于评估模式
    model.netCD.eval()
    
    # 准备评估容器
    preds = []
    labels = []
    process_times = []
    
    # 启动GPU监控
    if opt.gpu_ids:
        gpu_monitor = GPUMonitor(opt.gpu_ids[0])
        gpu_monitor.start()
        print(f"已启动GPU {opt.gpu_ids[0]} 监控...")
    else:
        gpu_monitor = None
        print("未使用GPU，性能监控将不可用")
    
    # 记录整体开始时间
    overall_start_time = time.time()
    
    print(f"开始处理 {len(test_loader)} 个测试样本...")
    
    # 在测试开始前清理CUDA缓存，确保测量准确
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 预热一次前向传播，避免第一次运行的额外开销影响测量
    if len(test_loader) > 0:
        with torch.no_grad():
            data = next(iter(test_loader))
            if len(data) == 5 and opt.load_t2_opt:
                model.set_input(data[0], data[1], data[2], data[3], opt.gpu_ids[0], data[4])
            else:
                model.set_input(data[0], data[1], data[2], data[3], opt.gpu_ids[0])
            _ = model.forward_CD()
            torch.cuda.synchronize()  # 确保GPU操作完成
    
    # 使用tqdm创建进度条
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, desc="测试进度")):
            # 限制测试数量
            if i >= opt.num_test:
                break
                
            # 记录每个样本的处理开始时间
            start_time = time.time()
            
            # 设置模型输入
            if len(data) == 5 and opt.load_t2_opt:  # 带时间点2的光学图像情况
                model.set_input(data[0], data[1], data[2], data[3], opt.gpu_ids[0], data[4])
            else:  # 不带时间点2的光学图像情况
                model.set_input(data[0], data[1], data[2], data[3], opt.gpu_ids[0])
            
            # 获取预测结果 - 使用forward_CD()方法而不是get_cd_pred()
            out_change = model.forward_CD()
            if isinstance(out_change, list):
                out_change = out_change[-1]  # 取最后一个输出
            
            # 确保GPU计算完成
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            # 调整大小确保一致
            out_change = F.interpolate(out_change, size=(512, 512), mode='bilinear', align_corners=True)
            
            # 获取预测标签
            pred = torch.argmax(out_change, dim=1)
            
            # 转换为numpy数组
            pred_np = pred.cpu().numpy().astype(np.uint8)
            label_np = data[2].cpu().numpy().astype(np.uint8)
            
            # 记录处理时间
            process_time = time.time() - start_time
            process_times.append(process_time)
            
            # 收集预测和标签
            preds.append(pred_np)
            labels.append(label_np)
            
            # 保存预测结果
            if opt.save_images:
                for b in range(pred_np.shape[0]):
                    name = data[3][b]
                    save_path = os.path.join(pred_dir, name)
                    cv2.imwrite(save_path, pred_np[b] * 255)
            
            # 生成并保存彩色比对图
            if opt.save_comparison:
                for b in range(pred_np.shape[0]):
                    name = data[3][b]
                    # 生成彩色比对图 - BGR转RGB (OpenCV默认BGR)
                    comp_img = generate_comparison_image(pred_np[b], label_np[b])
                    comp_img = cv2.cvtColor(comp_img, cv2.COLOR_RGB2BGR)  # 转为BGR以便使用OpenCV保存
                    
                    # 构建保存路径，使用原文件名但加上_comp后缀
                    base_name, ext = os.path.splitext(name)
                    comp_name = f"{base_name}_comp{ext}"
                    comp_path = os.path.join(comp_dir, comp_name)
                    
                    # 保存比对图
                    cv2.imwrite(comp_path, comp_img)
    
    # 计算总测试时间
    overall_time = time.time() - overall_start_time
    
    # 停止GPU监控
    if gpu_monitor:
        gpu_monitor.stop()
        gpu_stats = gpu_monitor.get_stats()
        gpu_monitor.join()  # 等待线程结束
    else:
        gpu_stats = {}
    
    # 计算平均处理时间
    avg_time = np.mean(process_times)
    total_time = np.sum(process_times)
    
    # 合并所有预测和标签
    preds_all = np.concatenate(preds, axis=0)
    labels_all = np.concatenate(labels, axis=0)
    
    # 计算混淆矩阵和评分
    hist = get_confuse_matrix(2, labels_all, preds_all)
    score = cm2score(hist)
    
    # 添加处理时间信息
    score['avg_process_time'] = avg_time
    score['total_process_time'] = total_time
    score['overall_test_time'] = overall_time
    score['samples_count'] = len(preds_all)
    
    # 添加GPU性能统计
    for key, value in gpu_stats.items():
        score[f'gpu_{key}'] = value
    
    # 格式化时间字符串，保留毫秒精度
    formatted_total_time = format_time(total_time)
    formatted_overall_time = format_time(overall_time)
    
    # 打印评估结果
    print('\n' + '='*50)
    print(f"测试完成! 共处理 {len(preds_all)} 个样本")
    print(f"总耗时: {formatted_total_time} (纯推理)")
    print(f"整体测试耗时: {formatted_overall_time} (包含数据加载和结果处理)")
    print(f"平均每样本处理时间: {avg_time:.6f} 秒")
    print(f"每秒处理样本数: {1/avg_time:.4f}")
    
    # 打印GPU性能统计
    if gpu_stats:
        print('\nGPU性能统计:')
        print(f"  内存使用率: 平均 {gpu_stats.get('memory_percent_avg', 0):.2f}%, 最大 {gpu_stats.get('memory_percent_max', 0):.2f}%")
        print(f"  内存占用: 平均 {gpu_stats.get('memory_used_avg', 0):.2f}MB / {gpu_stats.get('memory_total_avg', 0):.2f}MB")
        print(f"  GPU利用率: 平均 {gpu_stats.get('utilization_avg', 0):.2f}%, 最大 {gpu_stats.get('utilization_max', 0):.2f}%")
        print(f"  功耗: 平均 {gpu_stats.get('power_avg', 0):.2f}W, 最大 {gpu_stats.get('power_max', 0):.2f}W")
        print(f"  温度: 平均 {gpu_stats.get('temperature_avg', 0):.2f}°C, 最大 {gpu_stats.get('temperature_max', 0):.2f}°C")
        print(f"  PyTorch分配内存: 平均 {gpu_stats.get('torch_allocated_avg', 0):.2f}MB, 最大 {gpu_stats.get('torch_allocated_max', 0):.2f}MB")
        print(f"  PyTorch保留内存: 平均 {gpu_stats.get('torch_reserved_avg', 0):.2f}MB, 最大 {gpu_stats.get('torch_reserved_max', 0):.2f}MB")
    
    print('\n评估指标:')
    for key, value in {k: v for k, v in score.items() if not k.startswith('gpu_')}.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    print('='*50)
    
    # 创建简易结果摘要文件
    summary_path = os.path.join(result_dir, 'summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"变化检测测试摘要 - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n")
        f.write(f"{'='*50}\n")
        f.write(f"数据集: {opt.dataroot}\n")
        f.write(f"模型: {opt.model_path}\n")
        f.write(f"样本数量: {len(preds_all)}\n")
        f.write(f"总耗时: {formatted_total_time} (纯推理)\n")
        f.write(f"整体测试耗时: {formatted_overall_time} (包含数据加载和结果处理)\n")
        f.write(f"平均每样本处理时间: {avg_time:.6f} 秒\n")
        f.write(f"每秒处理样本数: {1/avg_time:.4f}\n\n")
        
        # 写入彩色比对图说明
        if opt.save_comparison:
            f.write("彩色比对图说明:\n")
            f.write("  白色 (255,255,255): 真正例 - 真实变化且预测为变化\n")
            f.write("  黑色 (0,0,0): 真负例 - 真实无变化且预测为无变化\n")
            f.write("  红色 (255,0,0): 漏检 - 真实变化但预测为无变化\n")
            f.write("  绿色 (0,255,0): 多检 - 真实无变化但预测为变化\n\n")
        
        # 写入GPU统计信息
        if gpu_stats:
            f.write("GPU性能统计:\n")
            f.write(f"  内存使用率: 平均 {gpu_stats.get('memory_percent_avg', 0):.2f}%, 最大 {gpu_stats.get('memory_percent_max', 0):.2f}%\n")
            f.write(f"  内存占用: 平均 {gpu_stats.get('memory_used_avg', 0):.2f}MB / {gpu_stats.get('memory_total_avg', 0):.2f}MB\n")
            f.write(f"  GPU利用率: 平均 {gpu_stats.get('utilization_avg', 0):.2f}%, 最大 {gpu_stats.get('utilization_max', 0):.2f}%\n")
            f.write(f"  功耗: 平均 {gpu_stats.get('power_avg', 0):.2f}W, 最大 {gpu_stats.get('power_max', 0):.2f}W\n")
            f.write(f"  温度: 平均 {gpu_stats.get('temperature_avg', 0):.2f}°C, 最大 {gpu_stats.get('temperature_max', 0):.2f}°C\n")
            f.write(f"  PyTorch分配内存: 平均 {gpu_stats.get('torch_allocated_avg', 0):.2f}MB, 最大 {gpu_stats.get('torch_allocated_max', 0):.2f}MB\n")
            f.write(f"  PyTorch保留内存: 平均 {gpu_stats.get('torch_reserved_avg', 0):.2f}MB, 最大 {gpu_stats.get('torch_reserved_max', 0):.2f}MB\n\n")
        
        f.write("评估指标:\n")
        for key, value in {k: v for k, v in score.items() if not k.startswith('gpu_')}.items():
            if isinstance(value, float):
                f.write(f"  {key}: {value:.6f}\n")
            else:
                f.write(f"  {key}: {value}\n")
    
    # 保存详细的性能监控数据
    if gpu_stats and gpu_monitor:
        # 创建性能监控目录
        perf_dir = os.path.join(result_dir, 'performance')
        os.makedirs(perf_dir, exist_ok=True)
        
        # 保存原始数据为CSV
        raw_data = {k: v for k, v in gpu_monitor.stats.items()}
        if raw_data:
            df = pd.DataFrame(raw_data)
            csv_path = os.path.join(perf_dir, 'gpu_performance_raw.csv')
            df.to_csv(csv_path, index=False)
            print(f"详细GPU性能数据已保存至: {csv_path}")
    
    print(f"测试摘要已保存至: {summary_path}")
    return score

if __name__ == '__main__':
    # 获取测试选项
    opt = TestOptions().parse()
    
    # 运行测试
    test_results = test_model(opt)