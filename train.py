import os
import time
import random
import numpy as np
import torch.nn as nn
import torch.autograd
from skimage import io
from torch import optim

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from options.train_options import TrainOptions
from models import create_model
from models.HeteGAN import Pix2PixModel
from utils.visualizer import Visualizer
from utils.util import accuracy, SCDD_eval_all, AverageMeter, get_confuse_matrix, cm2score

import torch.multiprocessing

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
    
    # 使用opt中的配置加载数据集
    train_set_change = dataset.Data('train', root=opt.dataroot, opt=opt)
    train_loader_change = DataLoader(train_set_change, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True,
                                     drop_last=True)
    dataset_size = len(train_loader_change)
    val_set = dataset.Data('val', root=opt.dataroot)
    val_loader = DataLoader(val_set, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False, drop_last=True)
    model = Pix2PixModel(opt, is_train=True)
    model.setup(opt)
    visualizer = Visualizer(opt)

    total_iters = 0
    resume_epoch = 0
    best_iou = 0
    for epoch in range(resume_epoch,
                       opt.n_epochs):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        preds_all = []
        labels_all = []
        names_all = []
        for i, data in enumerate(train_loader_change):
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data[0], data[1], data[2], data[3],
                            opt.gpu_ids[0])  # unpack data from dataset and apply preprocessing

            out_change = model.optimize_parameters(
                epoch)  # calculate loss functions, get gradients, update network weights
            out_change = FF.interpolate(out_change, size=(512, 512), mode='bilinear', align_corners=True)
            preds = torch.argmax(out_change, dim=1)
            pred_numpy = preds.cpu().numpy()
            labels_numpy = data[2].cpu().numpy()
            preds_all.append(pred_numpy)
            labels_all.append(labels_numpy)
            names_all.extend(data[3])

            if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                # if opt.display_id > 0:
                #     visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            iter_data_time = time.time()
        preds_all = np.concatenate(preds_all, axis=0)
        labels_all = np.concatenate(labels_all, axis=0)
        hist = get_confuse_matrix(2, labels_all, preds_all)
        score = cm2score(hist)
        # with open(os.path.join(opt.checkpoints_dir,opt.name,"cd_log.txt"), 'a') as f:
        #     f.write('Epoch: %d  score: %s\n'\
        #         %(epoch,{key: score[key] for key in score}))
        print('训练轮次: %d 评分: %s' % (epoch, {key: score[key] for key in score}))

        # 记录训练结果
        train_score = score
        train_iou = score['iou_1']  # 保存训练集上的iou_1

        # 获取训练损失
        train_losses = model.get_current_losses()
        train_loss = sum(train_losses.values()) if train_losses else 0

        best_preds_dir = os.path.join(opt.checkpoints_dir, opt.name, "results")
        if not os.path.exists(best_preds_dir):
            os.makedirs(best_preds_dir)
        val_loss = AverageMeter()
        preds_all = []
        labels_all = []
        names_all = []
        for i, data in enumerate(val_loader):
            model.set_input(data[0], data[1], data[2], data[3], opt.gpu_ids[0])
            out_change, loss = model.get_val_pred()
            out_change = FF.interpolate(out_change, size=(512, 512), mode='bilinear', align_corners=True)
            val_loss.update(loss.cpu().detach().numpy())
            preds = torch.argmax(out_change, dim=1)
            pred_numpy = preds.cpu().numpy()
            labels_numpy = data[2].cpu().numpy()
            preds_all.append(pred_numpy)
            labels_all.append(labels_numpy)
            names_all.extend(data[3])
        preds_all = np.concatenate(preds_all, axis=0)
        labels_all = np.concatenate(labels_all, axis=0)
        hist = get_confuse_matrix(2, labels_all, preds_all)
        score = cm2score(hist)
        model.save_networks(epoch)
        if score['iou_1'] > best_iou:
            best_iou = score['iou_1']
            model.save_networks(epoch, save_best=True)
            for i in range(len(names_all)):
                save_path = os.path.join(best_preds_dir, names_all[i])
                cv2.imwrite(save_path, preds_all[i] * 255)
            print('更新最佳IoU模型')
        with open(os.path.join(opt.checkpoints_dir, opt.name, "cd_log.txt"), 'a') as f:
            # 添加分隔行
            f.write('='*100 + '\n')
            # 合并展示训练和验证结果
            f.write('【Epoch: %d】训练IoU: %.4f (Loss: %.4f) | 验证IoU: %.4f/%.4f (Loss: %.4f)\n' %
                   (epoch, train_iou, train_loss, score['iou_1'], best_iou, val_loss.average()))

            # 对比展示关键指标 - 使用固定宽度确保对齐
            f.write('╔═════════╦═══════════════╦═══════════════╦═══════════════╗\n')
            f.write('║ 指标对比 ║     准确率     ║    平均IoU     ║    平均F1      ║\n')
            f.write('╠═════════╬═══════════════╬═══════════════╬═══════════════╣\n')
            f.write('║  训练集  ║     %-7.4f   ║     %-7.4f   ║     %-7.4f   ║\n' %
                   (train_score['acc'], train_score['miou'], train_score['mf1']))
            f.write('║  验证集  ║     %-7.4f   ║     %-7.4f   ║     %-7.4f   ║\n' %
                   (score['acc'], score['miou'], score['mf1']))
            f.write('╚═════════╩═══════════════╩═══════════════╩═══════════════╝\n')

            # 分别记录详细指标
            f.write('训练详细指标: %s\n' % {k: round(v, 4) if isinstance(v, float) else v for k, v in train_score.items()})
            f.write('验证详细指标: %s\n' % {k: round(v, 4) if isinstance(v, float) else v for k, v in score.items()})
            f.write('='*100 + '\n\n')

        # 美化控制台输出
        print('='*100)
        # 合并展示训练和验证结果
        print('【Epoch: %d】训练IoU: %.4f (Loss: %.4f) | 验证IoU: %.4f/%.4f (Loss: %.4f)' %
             (epoch, train_iou, train_loss, score['iou_1'], best_iou, val_loss.average()))

        # 对比展示关键指标 - 使用固定宽度确保对齐
        print('╔═════════╦═══════════════╦═══════════════╦═══════════════╗')
        print('║ 指标对比 ║     准确率     ║    平均IoU     ║    平均F1      ║')
        print('╠═════════╬═══════════════╬═══════════════╬═══════════════╣')
        print('║  训练集  ║     %-7.4f   ║     %-7.4f   ║     %-7.4f   ║' %
             (train_score['acc'], train_score['miou'], train_score['mf1']))
        print('║  验证集  ║     %-7.4f   ║     %-7.4f   ║     %-7.4f   ║' %
             (score['acc'], score['miou'], score['mf1']))
        print('╚═════════╩═══════════════╩═══════════════╩═══════════════╝')

        # 如果验证集IoU优于之前最佳值，显示提示
        if score['iou_1'] > best_iou - 0.0001:  # 考虑浮点精度
            print('🌟 本轮验证IoU创建新高！')

        print('训练轮次 %d / %d 结束 \t 耗时: %d 秒' % (epoch, opt.n_epochs, time.time() - epoch_start_time))

        # 在每个epoch结束时更新学习率
        model.update_learning_rate()
