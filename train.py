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
# 生成一个随机5位整数
import math

nums = math.floor(1e5 * random.random())
seed = 666


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


#  torch.backends.cudnn.deterministic = True
# # 设置随机数种子
setup_seed(seed)
# from models.SSCDl import SSCDl as Net

###############################################    
# Training options
###############################################

if __name__ == '__main__':
    opt = TrainOptions().parse()  # get training options
    # transform_params = get_params(opt, (512,512))
    # img_transform = get_transform(opt, transform_params, grayscale=(3 == 1))
    # B_transform = get_transform(opt, transform_params, grayscale=(3 == 1))
    # train_set_change = dataset.Data('train', root=opt.dataroot)
    train_set_change = dataset.Data('train', root=None, opt=opt)
    train_loader_change = DataLoader(train_set_change, batch_size=opt.batch_size, num_workers=8, shuffle=True,
                                     drop_last=True)
    dataset_size = len(train_loader_change)
    val_set = dataset.Data('val', root=opt.dataroot)
    val_loader = DataLoader(val_set, batch_size=opt.batch_size, num_workers=8, shuffle=False, drop_last=True)
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
        model.update_learning_rate()  # update learning rates in the beginning of every epoch.
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
        print('Epoch: %d  score: %s' % (epoch, {key: score[key] for key in score}))

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
            print('update best iou model')
        with open(os.path.join(opt.checkpoints_dir, opt.name, "cd_log.txt"), 'a') as f:
            f.write('Epoch: %d  best_iou: %.2f  Val loss: %.2f  score: %s\n' % (epoch, best_iou, val_loss.average(),
                                                                                {key: score[key] for key in score}))
        print('Epoch: %d  best_iou: %.2f  Val loss: %.2f  score: %s\n' % (epoch, best_iou, val_loss.average(),
                                                                          {key: score[key] for key in score}))
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs, time.time() - epoch_start_time))
