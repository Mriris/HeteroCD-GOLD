import time

import cv2
import torch.autograd
import torch.multiprocessing
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from models.HeteCD import TripleHeteCD
from options.train_options import TrainOptions
from utils.util import AverageMeter, get_confuse_matrix, cm2score
from utils.visualizer import Visualizer

torch.multiprocessing.set_sharing_strategy('file_system')

# Data and model choose
torch.set_num_threads(4)
import torch.nn.functional as FF
###############################################
from datasets import dataset
from datasets.dataset import *

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

###############################################    
# Training options
###############################################

if __name__ == '__main__':
    opt = TrainOptions().parse()  # get training options
    
    # 设置随机数种子
    setup_seed(opt.seed)
    
    # 使用opt中的配置加载数据集，设置load_t2_opt=True来加载时间点2的光学图像
    train_set_change = dataset.Data('train', root=opt.dataroot, opt=opt, load_t2_opt=True)
    train_loader_change = DataLoader(train_set_change, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True,
                                     drop_last=True)
    dataset_size = len(train_loader_change)
    
    # 加载验证集，同样加载时间点2的光学图像
    val_set = dataset.Data('val', root=opt.dataroot, load_t2_opt=True)
    val_loader = DataLoader(val_set, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False, drop_last=True)
    
    # 创建模型，设置use_distill=True启用蒸馏学习
    opt.use_distill = True  # 启用蒸馏学习
    model = TripleHeteCD(opt, is_train=True)
    
    # 添加断点续训功能：恢复之前的训练状态
    resume_epoch = 0
    best_iou = 0
    
    if opt.continue_train:
        # 读取之前的训练记录
        training_info_path = os.path.join(opt.checkpoints_dir, opt.name, "training_info.txt")
        
        # 检查是否存在训练信息文件
        if os.path.exists(training_info_path):
            with open(training_info_path, 'r') as f:
                lines = f.readlines()
                training_info = {}
                
                # 解析训练信息文件中的所有键值对
                for line in lines:
                    if ':' in line:
                        key, value = line.strip().split(':', 1)
                        training_info[key] = value
                
                try:
                    # 获取下一个要训练的epoch
                    if 'next_epoch' in training_info:
                        resume_epoch = int(training_info['next_epoch'])
                    elif 'current_epoch' in training_info:
                        # 如果没有next_epoch，则从current_epoch + 1开始
                        resume_epoch = int(training_info['current_epoch']) + 1
                    
                    # 获取历史最佳IoU
                    if 'best_iou' in training_info:
                        best_iou = float(training_info['best_iou'])
                    
                    print(f"断点续训：从轮次 {resume_epoch} 继续训练，历史最佳IoU: {best_iou:.4f}")
                    
                    # 确定要加载的模型文件
                    # 优先使用指定的latest_model
                    if 'latest_model' in training_info:
                        opt.epoch = training_info['latest_model'].replace('_net_CD.pth', '')
                        print(f"将加载最新模型：{opt.epoch}")
                    else:
                        # 否则使用current_epoch作为加载点
                        if 'current_epoch' in training_info:
                            opt.epoch = training_info['current_epoch']
                            print(f"将加载指定epoch模型：{opt.epoch}")
                        else:
                            # 没有明确指定，则寻找最新的模型
                            opt.epoch = 'latest'
                            print("将尝试加载最新模型")
                except Exception as e:
                    print(f"解析训练信息文件出错: {e}")
                    print("训练信息文件格式错误，将从头开始训练")
                    resume_epoch = 0
                    best_iou = 0
                    opt.epoch = 'latest'
        else:
            print("未找到训练信息文件，将从头开始训练")
    
    # 设置模型，这将加载保存的模型权重（如果存在）
    model.setup(opt)
    visualizer = Visualizer(opt)
    
    # 如果是断点续训，输出确认信息
    if opt.continue_train and resume_epoch > 0:
        print(f"模型已从{opt.epoch}加载，将从epoch {resume_epoch}继续训练")
        # 检查模型是否正确加载
        param_sum = sum(p.sum().item() for p in model.netCD.parameters() if p.requires_grad)
        print(f"模型参数总和: {param_sum:.4f} - {'正常' if abs(param_sum) > 0.1 else '警告: 可能未正确加载'}")
        
        # 如果模型参数异常，给出更详细的警告
        if abs(param_sum) <= 0.1:
            print("警告：模型参数总和接近零，这可能表明模型未正确加载！")
            print("请检查模型文件路径和权重是否正确。")
            response = input("模型参数可能有问题，是否继续训练？(y/n): ")
            if response.lower() != 'y':
                print("训练已取消")
                exit(0)
    
    # 创建TensorBoard摘要写入器
    log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'tensorboard_logs')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # 在TensorBoard中记录训练参数
    opt_dict = vars(opt)
    for k in sorted(opt_dict.keys()):
        if isinstance(opt_dict[k], (int, float, str, bool)):
            writer.add_text('Parameters/' + k, str(opt_dict[k]), 0)

    # 创建日志目录
    if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.name)):
        os.makedirs(os.path.join(opt.checkpoints_dir, opt.name))
    
    # 日志文件路径
    log_path = os.path.join(opt.checkpoints_dir, opt.name, "cd_log.txt")
    
    # 如果是继续训练且日志文件不存在，或者是新训练，则创建新的日志文件
    if (not opt.continue_train) or (opt.continue_train and not os.path.exists(log_path)):
        with open(log_path, 'w') as f:
            f.write(f"HeteGAN 训练日志 - 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据集: {opt.dataroot}\n")
            f.write(f"批次大小: {opt.batch_size}, 学习率: {opt.lr}, GPU: {opt.gpu_ids}\n")
            f.write(f"是否使用蒸馏学习: {'是' if opt.use_distill else '否'}\n")
            
            # 添加所有训练参数记录
            f.write("\n=== 训练参数 ===\n")
            # 获取opt中的所有属性并排序
            opt_dict = vars(opt)
            for k in sorted(opt_dict.keys()):
                f.write(f"{k}: {opt_dict[k]}\n")
            f.write("=== 参数结束 ===\n\n")
            
            f.write("─" * 50 + "\n")

    total_iters = 0
    
    # 如果是继续训练，记录续训信息到日志文件
    if opt.continue_train and resume_epoch > 0:
        with open(log_path, 'a') as f:
            f.write('─' * 50 + f'\n断点续训：从轮次 {resume_epoch} 继续训练，历史最佳IoU: {best_iou:.4f}\n' + '─' * 50 + '\n')

    # 尝试使用混合精度训练，但默认为False避免出现NaN值
    use_amp = opt.use_amp if hasattr(opt, 'use_amp') else False
    try:
        from torch.cuda.amp import GradScaler, autocast
        if use_amp:
            scaler = GradScaler()
            print("启用混合精度训练 (AMP)")
        else:
            print("混合精度训练可用但未启用。如需启用，请使用 --use_amp 选项")
    except ImportError:
        use_amp = False
        print("混合精度训练不可用 - 需要PyTorch >= 1.6")
    
    # 获取梯度裁剪参数
    gradient_clip_norm = opt.gradient_clip_norm if hasattr(opt, 'gradient_clip_norm') else 1.0
    
    for epoch in range(resume_epoch,
                       opt.n_epochs):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        
        # 设置当前轮次到模型中，用于动态权重计算
        if hasattr(model, 'set_epoch'):
            model.set_epoch(epoch)
        
        # 打印当前轮次开始信息
        print('\n' + '=' * 80)
        print(f'开始训练第 {epoch}/{opt.n_epochs - 1} 轮 | 批次大小: {opt.batch_size} | 学习率: {opt.lr}')
        print('=' * 80)
        
        # 学生网络的预测结果和标签
        preds_all = []
        labels_all = []
        names_all = []
        
        # 教师网络的预测结果和标签
        teacher_preds_all = []
        teacher_labels_all = []
        
        for i, data in enumerate(train_loader_change):
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            
            # 设置模型输入，现在包括时间点2的光学图像
            if len(data) == 5:  # 带时间点2的光学图像的情况
                model.set_input(data[0], data[1], data[2], data[3], opt.gpu_ids[0], data[4])
            else:  # 不带时间点2的光学图像的情况
                model.set_input(data[0], data[1], data[2], data[3], opt.gpu_ids[0])

            # 优化模型参数
            model.optimizer_G.zero_grad()  # 清空梯度缓存
            
            if use_amp:
                # 使用混合精度进行前向和反向传播
                with autocast():
                    out_change = model.forward_CD()  # 前向传播
                    model.compute_losses()  # 计算损失但不立即反向传播
                
                # 使用缩放器缩放损失值，避免数值下溢
                scaler.scale(model.loss_G).backward()
                
                # 梯度裁剪避免梯度爆炸
                scaler.unscale_(model.optimizer_G)
                torch.nn.utils.clip_grad_norm_(model.netCD.parameters(), max_norm=gradient_clip_norm)
                
                # 更新权重
                scaler.step(model.optimizer_G)
                scaler.update()
            else:
                # 常规训练流程
                out_change = model.optimize_parameters(epoch)
            
            # 插值到统一大小
            out_change = FF.interpolate(out_change, size=(512, 512), mode='bilinear', align_corners=True)
            
            # 记录学生网络的预测结果
            with torch.no_grad():
                preds = torch.argmax(out_change, dim=1)
                pred_numpy = preds.cpu().numpy()
                labels_numpy = data[2].cpu().numpy()
                preds_all.append(pred_numpy)
                labels_all.append(labels_numpy)
                names_all.extend(data[3])
            
            # 如果使用蒸馏学习且有第三张图，也记录教师网络的预测结果
            if opt.use_distill and len(data) == 5:
                with torch.no_grad():
                    teacher_pred, _ = model.get_teacher_pred()
                    if teacher_pred is not None:
                        teacher_pred = FF.interpolate(teacher_pred, size=(512, 512), mode='bilinear', align_corners=True)
                        teacher_preds = torch.argmax(teacher_pred, dim=1)
                        teacher_pred_numpy = teacher_preds.cpu().numpy()
                        teacher_preds_all.append(teacher_pred_numpy)
                        teacher_labels_all.append(labels_numpy)  # 标签是相同的

            if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                # 汉化输出格式
                loss_str = ' '.join([f'{name}: {value:.3f}' for name, value in losses.items()])
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                print(f'(轮次: {epoch}, 批次: {i}, 用时: {t_comp:.3f}秒/样本, 数据加载: {t_data:.3f}秒) {loss_str}')

            iter_data_time = time.time()
            
        # 评估学生网络在训练集上的性能
        preds_all = np.concatenate(preds_all, axis=0)
        labels_all = np.concatenate(labels_all, axis=0)
        hist = get_confuse_matrix(2, labels_all, preds_all)
        score = cm2score(hist)
        
        # 记录训练指标到TensorBoard
        writer.add_scalar('Train/Student/Accuracy', score['acc'], epoch)
        writer.add_scalar('Train/Student/MeanIoU', score['miou'], epoch)
        writer.add_scalar('Train/Student/IoU_0', score['iou_0'], epoch)
        writer.add_scalar('Train/Student/IoU_1', score['iou_1'], epoch)
        writer.add_scalar('Train/Student/F1_0', score['F1_0'], epoch)
        writer.add_scalar('Train/Student/F1_1', score['F1_1'], epoch)
        
        # 记录训练损失
        for loss_name, loss_value in model.get_current_losses().items():
            writer.add_scalar(f'Train/Loss_{loss_name}', loss_value, epoch)
            
        print('学生网络训练评分: %s' % {key: score[key] for key in score})
        
        # 记录训练结果
        train_score = score
        train_iou = score['iou_1']  # 保存训练集上的iou_1
        
        # 记录教师网络在训练集上的性能（如果有）
        teacher_score = None
        if opt.use_distill and len(teacher_preds_all) > 0:
            teacher_preds_all = np.concatenate(teacher_preds_all, axis=0)
            teacher_labels_all = np.concatenate(teacher_labels_all, axis=0)
            hist = get_confuse_matrix(2, teacher_labels_all, teacher_preds_all)
            teacher_score = cm2score(hist)
            
            # 记录教师网络训练指标到TensorBoard
            writer.add_scalar('Train/Teacher/Accuracy', teacher_score['acc'], epoch)
            writer.add_scalar('Train/Teacher/MeanIoU', teacher_score['miou'], epoch)
            writer.add_scalar('Train/Teacher/IoU_0', teacher_score['iou_0'], epoch)
            writer.add_scalar('Train/Teacher/IoU_1', teacher_score['iou_1'], epoch)
            writer.add_scalar('Train/Teacher/F1_0', teacher_score['F1_0'], epoch)
            writer.add_scalar('Train/Teacher/F1_1', teacher_score['F1_1'], epoch)
            
            print('教师网络训练评分: %s' % {key: teacher_score[key] for key in teacher_score})

        # 获取训练损失
        train_losses = model.get_current_losses()
        train_loss = sum(train_losses.values()) if train_losses else 0

        # 创建保存预测结果的目录
        best_preds_dir = os.path.join(opt.checkpoints_dir, opt.name, "results")
        if not os.path.exists(best_preds_dir):
            os.makedirs(best_preds_dir)
            
        # 验证集评估
        val_loss = AverageMeter()
        preds_all_val = []
        labels_all_val = []
        names_all_val = []
        
        # 教师网络的验证集预测
        teacher_preds_all_val = []
        teacher_labels_all_val = []
        teacher_val_loss = AverageMeter()
        
        # 设置为评估模式
        model.netCD.eval()
        
        for i, data in enumerate(val_loader):
            # 设置模型输入，现在包括时间点2的光学图像
            if len(data) == 5:  # 带时间点2的光学图像的情况
                model.set_input(data[0], data[1], data[2], data[3], opt.gpu_ids[0], data[4])
            else:  # 不带时间点2的光学图像的情况
                model.set_input(data[0], data[1], data[2], data[3], opt.gpu_ids[0])
                
            # 获取学生网络预测结果
            with torch.no_grad():
                out_change, loss = model.get_val_pred()
                out_change = FF.interpolate(out_change, size=(512, 512), mode='bilinear', align_corners=True)
                val_loss.update(loss.cpu().detach().numpy())
                preds = torch.argmax(out_change, dim=1)
                pred_numpy = preds.cpu().numpy()
                labels_numpy = data[2].cpu().numpy()
                preds_all_val.append(pred_numpy)
                labels_all_val.append(labels_numpy)
                names_all_val.extend(data[3])
                
                # 如果使用蒸馏学习且有第三张图像，获取教师网络预测结果
                if opt.use_distill and len(data) == 5:
                    teacher_pred, teacher_loss = model.get_teacher_pred()
                    if teacher_pred is not None:
                        teacher_pred = FF.interpolate(teacher_pred, size=(512, 512), mode='bilinear', align_corners=True)
                        teacher_val_loss.update(teacher_loss.cpu().detach().numpy())
                        teacher_preds = torch.argmax(teacher_pred, dim=1)
                        teacher_pred_numpy = teacher_preds.cpu().numpy()
                        teacher_preds_all_val.append(teacher_pred_numpy)
                        teacher_labels_all_val.append(labels_numpy)  # 标签是相同的
        
        # 评估学生网络在验证集上的性能
        preds_all_val = np.concatenate(preds_all_val, axis=0)
        labels_all_val = np.concatenate(labels_all_val, axis=0)
        hist = get_confuse_matrix(2, labels_all_val, preds_all_val)
        score = cm2score(hist)
        
        # 记录验证指标到TensorBoard
        writer.add_scalar('Validation/Student/Accuracy', score['acc'], epoch)
        writer.add_scalar('Validation/Student/MeanIoU', score['miou'], epoch)
        writer.add_scalar('Validation/Student/IoU_0', score['iou_0'], epoch)
        writer.add_scalar('Validation/Student/IoU_1', score['iou_1'], epoch)
        writer.add_scalar('Validation/Student/F1_0', score['F1_0'], epoch)
        writer.add_scalar('Validation/Student/F1_1', score['F1_1'], epoch)
        writer.add_scalar('Validation/Student/Loss', val_loss.average(), epoch)
        
        # 记录验证集上的IoU，用于评估模型性能
        val_iou = score['iou_1']
        
        # 评估教师网络在验证集上的性能（如果有）
        teacher_val_score = None
        if opt.use_distill and len(teacher_preds_all_val) > 0:
            teacher_preds_all_val = np.concatenate(teacher_preds_all_val, axis=0)
            teacher_labels_all_val = np.concatenate(teacher_labels_all_val, axis=0)
            hist = get_confuse_matrix(2, teacher_labels_all_val, teacher_preds_all_val)
            teacher_val_score = cm2score(hist)
            
            # 记录教师网络验证指标到TensorBoard
            writer.add_scalar('Validation/Teacher/Accuracy', teacher_val_score['acc'], epoch)
            writer.add_scalar('Validation/Teacher/MeanIoU', teacher_val_score['miou'], epoch)
            writer.add_scalar('Validation/Teacher/IoU_0', teacher_val_score['iou_0'], epoch)
            writer.add_scalar('Validation/Teacher/IoU_1', teacher_val_score['iou_1'], epoch)
            writer.add_scalar('Validation/Teacher/F1_0', teacher_val_score['F1_0'], epoch)
            writer.add_scalar('Validation/Teacher/F1_1', teacher_val_score['F1_1'], epoch)
            writer.add_scalar('Validation/Teacher/Loss', teacher_val_loss.average(), epoch)
            
            print('教师网络验证评分: %s' % {key: teacher_val_score[key] for key in teacher_val_score})
        
        # 保存模型前清理旧的非最佳模型文件，只保留最新和最好的
        model_dir = os.path.join(opt.checkpoints_dir, opt.name)
        # 找出当前目录下所有模型文件
        model_files = [f for f in os.listdir(model_dir) 
                      if f.endswith('_net_CD.pth') and not f.startswith('best')]
        # 按照epoch编号排序
        if len(model_files) > 1:  # 如果有多个非最佳模型文件
            # 排序模型文件，保留最新的，删除其他的
            model_files.sort(key=lambda x: int(x.split('_')[0]) if x.split('_')[0].isdigit() else -1, reverse=True)
            latest_model = model_files[0]  # 保留最新的模型
            for model_file in model_files[1:]:  # 删除其他旧模型
                try:
                    os.remove(os.path.join(model_dir, model_file))
                    print(f"已删除旧模型文件: {model_file}")
                except Exception as e:
                    print(f"删除文件 {model_file} 时出错: {e}")

        # 保存当前模型作为最新模型
        model.save_networks(epoch)
        print(f"已保存最新模型: {epoch}_net_CD.pth")
        latest_model_file = f"{epoch}_net_CD.pth"
        
        # 是否需要保存最佳模型
        best_model_updated = False
        if val_iou > best_iou:
            with open(os.path.join(opt.checkpoints_dir, opt.name, "cd_log.txt"), 'a') as f:
                f.write(f'新纪录！保存模型至: {os.path.join(opt.checkpoints_dir, opt.name)}\n')
            
            # 查找并删除之前的最佳模型文件
            for file in os.listdir(model_dir):
                if file.endswith('.pth') and file.startswith('best_net'):
                    try:
                        os.remove(os.path.join(model_dir, file))
                        print(f"已删除旧的最佳模型: {file}")
                    except Exception as e:
                        print(f"删除文件 {file} 时出错: {e}")
            
            # 保存新的最佳模型
            model.save_networks('best')
            best_iou = val_iou
            best_epoch = epoch  # 记录产生最佳结果的epoch
            best_model_file = f"best_net_CD.pth"
            best_model_updated = True
            
            # 保存最佳结果预测图
            for i in range(len(names_all_val)):
                save_path = os.path.join(best_preds_dir, names_all_val[i])
                cv2.imwrite(save_path, preds_all_val[i] * 255)
            
            print('🌟 更新最佳IoU模型 🌟')
        
        # 更新训练信息文件，确保信息完整
        with open(os.path.join(opt.checkpoints_dir, opt.name, "training_info.txt"), 'w') as f:
            # 下一个要训练的epoch
            next_epoch = epoch + 1
            f.write(f"next_epoch:{next_epoch}\n")
            # 当前已完成的epoch
            f.write(f"current_epoch:{epoch}\n")
            # 最佳性能及对应epoch
            f.write(f"best_iou:{best_iou}\n")
            if 'best_epoch' in locals():
                f.write(f"best_epoch:{best_epoch}\n")
            # 最新模型文件名
            f.write(f"latest_model:{latest_model_file}\n")
            # 最佳模型文件名
            f.write(f"best_model:best_net_CD.pth\n")
            # 记录最后一次更新最佳模型的轮次
            if best_model_updated:
                f.write(f"best_model_updated_at:{epoch}\n")
            
        with open(os.path.join(opt.checkpoints_dir, opt.name, "cd_log.txt"), 'a') as f:
            # 添加分隔行
            f.write('='*100 + '\n【Epoch: %d】\n' % epoch)
            
            # 如果使用动态权重，记录当前权重值
            if opt.use_dynamic_weights and hasattr(model, 'get_dynamic_weights'):
                model.set_epoch(epoch)  # 确保模型知道当前epoch
                cd_weight, distill_weight, diff_att_weight = model.get_dynamic_weights()
                f.write(f'【动态权重】CD损失: {cd_weight:.4f}, 蒸馏损失: {distill_weight:.4f}, 差异图注意力损失: {diff_att_weight:.4f}\n')
            
            # 合并展示训练和验证结果
            f.write('【学生网络】 - 训练IoU: %.4f (Loss: %.4f) | 验证IoU: %.4f/%.4f (Loss: %.4f)\n' %
                   (train_iou, train_loss, score['iou_1'], best_iou, val_loss.average()))
                   
            # 如果有教师网络，记录教师网络结果
            if teacher_score is not None and teacher_val_score is not None:
                f.write('【教师网络】 - 训练IoU: %.4f | 验证IoU: %.4f (Loss: %.4f)\n' %
                       (teacher_score['iou_1'], teacher_val_score['iou_1'], teacher_val_loss.average()))

            # 分别记录详细指标
            f.write('学生网络训练详细指标: %s\n' % {k: round(v, 4) if isinstance(v, float) else v for k, v in train_score.items()})
            f.write('学生网络验证详细指标: %s\n' % {k: round(v, 4) if isinstance(v, float) else v for k, v in score.items()})
            
            # 如果有教师网络，记录教师网络详细指标
            if teacher_score is not None and teacher_val_score is not None:
                f.write('教师网络训练详细指标: %s\n' % {k: round(v, 4) if isinstance(v, float) else v for k, v in teacher_score.items()})
                f.write('教师网络验证详细指标: %s\n' % {k: round(v, 4) if isinstance(v, float) else v for k, v in teacher_val_score.items()})

        # 美化控制台输出
        print('='*100)
        # 如果使用动态权重，打印当前权重值
        if opt.use_dynamic_weights and hasattr(model, 'get_dynamic_weights'):
            model.set_epoch(epoch)  # 确保模型知道当前epoch
            cd_weight, distill_weight, diff_att_weight = model.get_dynamic_weights()
            print(f'【动态权重】CD损失: {cd_weight:.4f}, 蒸馏损失: {distill_weight:.4f}, 差异图注意力损失: {diff_att_weight:.4f}')

        # 合并展示训练和验证结果
        print('【Epoch: %d】学生网络 - 训练IoU: %.4f (Loss: %.4f) | 验证IoU: %.4f/%.4f (Loss: %.4f)' %
             (epoch, train_iou, train_loss, score['iou_1'], best_iou, val_loss.average()))
             
        # 如果有教师网络，打印教师网络结果
        if teacher_score is not None and teacher_val_score is not None:
            print('【Epoch: %d】教师网络 - 训练IoU: %.4f | 验证IoU: %.4f (Loss: %.4f)' %
                 (epoch, teacher_score['iou_1'], teacher_val_score['iou_1'], teacher_val_loss.average()))

        # 对比展示关键指标 - 使用固定宽度确保对齐
        print('╔═════════╦════════════╦═══════════════╦═══════════════╦═══════════════╗')
        print('║ 网络类型 ║    节点    ║     准确率     ║    平均IoU     ║    平均F1      ║')
        print('╠═════════╬════════════╬═══════════════╬═══════════════╬═══════════════╣')
        print('║  学生网络 ║   训练集   ║     %-7.4f   ║     %-7.4f   ║     %-7.4f   ║' %
             (train_score['acc'], train_score['miou'], train_score['mf1']))
        print('║  学生网络 ║   验证集   ║     %-7.4f   ║     %-7.4f   ║     %-7.4f   ║' %
             (score['acc'], score['miou'], score['mf1']))
             
        # 如果有教师网络，打印教师网络对比
        if teacher_score is not None and teacher_val_score is not None:
            print('║  教师网络 ║   训练集   ║     %-7.4f   ║     %-7.4f   ║     %-7.4f   ║' %
                 (teacher_score['acc'], teacher_score['miou'], teacher_score['mf1']))
            print('║  教师网络 ║   验证集   ║     %-7.4f   ║     %-7.4f   ║     %-7.4f   ║' %
                 (teacher_val_score['acc'], teacher_val_score['miou'], teacher_val_score['mf1']))
        print('╚═════════╩════════════╩═══════════════╩═══════════════╩═══════════════╝')

        # 如果验证集IoU优于之前最佳值，显示提示
        if score['iou_1'] >= best_iou - 0.0001:  # 考虑浮点精度
            print('🌟 本轮验证IoU创建新高！')

        print('训练轮次 %d / %d 结束 \t 耗时: %d 秒' % (epoch, opt.n_epochs, time.time() - epoch_start_time))

        # 在每个epoch结束时更新学习率
        model.update_learning_rate()
        
    # 关闭TensorBoard写入器
    writer.close()
