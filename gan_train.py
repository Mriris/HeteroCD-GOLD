import os
import time

import torch.autograd
from torch.utils.data import DataLoader

from models.GOLD import TripleHeteCD
from options.train_options import TrainOptions
from utils.util import AverageMeter, get_confuse_matrix, cm2score
from utils.visualizer import Visualizer

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
#Data and model choose
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
#Training options
###############################################

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    transform_params = get_params(opt, (512,512))
    img_transform = get_transform(opt, transform_params, grayscale=(3 == 1))
    # B_transform = get_transform(opt, transform_params, grayscale=(3 == 1))
    train_set_change = dataset.Data('train',img_transform, root = opt.dataroot)
    train_loader_change = DataLoader(train_set_change, batch_size=opt.batch_size, num_workers=8, shuffle=True,drop_last=True)
    dataset_size = len(train_loader_change)
    val_set = dataset.Data('val',img_transform, root = opt.dataroot)
    val_loader = DataLoader(val_set, batch_size=opt.batch_size, num_workers=8, shuffle=False,drop_last=True)
    model = TripleHeteCD(opt, is_train=True)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_iters = 0 
    opt.n_epochs_gen = 0
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1): 
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        preds_all = []
        labels_all = []
        names_all = []
        for i, data in enumerate(train_loader_change):
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data[0],data[1],data[2],data[3],opt.gpu_ids[0])         # unpack data from dataset and apply preprocessing
            
            out_change = model.optimize_parameters(epoch)   # calculate loss functions, get gradients, update network weights
            out_change = FF.interpolate(out_change, size=(512,512), mode='bilinear', align_corners=True)
            preds = torch.argmax(out_change, dim=1)
            pred_numpy = preds.cpu().numpy()
            labels_numpy = data[2].cpu().numpy()
            preds_all.append(pred_numpy)
            labels_all.append(labels_numpy)
            names_all.extend(data[3])
            # print(cd_pred)
            if epoch < opt.n_epochs_gen:
                if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                    save_result = total_iters % opt.update_html_freq == 0
                    model.compute_visuals()
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                if epoch <= opt.n_epochs_gen:
                    losses = model.get_current_losses()
                else:
                    losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                # if opt.display_id > 0:
                #     visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
            iter_data_time = time.time()
        preds_all = np.concatenate(preds_all, axis=0)
        labels_all = np.concatenate(labels_all, axis=0)
        hist = get_confuse_matrix(2,labels_all,preds_all)
        score = cm2score(hist)
        # with open(os.path.join(opt.checkpoints_dir,opt.name,"cd_log.txt"), 'a') as f:
        #     f.write('Epoch: %d  score: %s\n'\
        #         %(epoch,{key: score[key] for key in score}))
        print('Epoch: %d  score: %s'\
        %(epoch,{key: score[key] for key in score}))
            
        if epoch>=opt.n_epochs_gen:
            val_loss = AverageMeter()
            preds_all = []
            labels_all = []
            names_all = []
            for i, data in enumerate(val_loader):
                model.set_input(data[0],data[1],data[2],data[3],opt.gpu_ids[0]) 
                out_change,loss = model.get_val_pred()
                out_change = FF.interpolate(out_change, size=(512,512), mode='bilinear', align_corners=True)
                
                val_loss.update(loss.cpu().detach().numpy())
                preds = torch.argmax(out_change, dim=1)
                pred_numpy = preds.cpu().numpy()
                labels_numpy = data[2].cpu().numpy()
                preds_all.append(pred_numpy)
                labels_all.append(labels_numpy)
                names_all.extend(data[3])
            preds_all = np.concatenate(preds_all, axis=0)
            labels_all = np.concatenate(labels_all, axis=0)
            hist = get_confuse_matrix(2,labels_all,preds_all)
            score = cm2score(hist)
            with open(os.path.join(opt.checkpoints_dir,opt.name,"cd_log.txt"), 'a') as f:
                f.write('Epoch: %d  Val loss: %.2f  score: %s\n'\
                    %(epoch, val_loss.average(),{key: score[key] for key in score}))
            print('Epoch: %d Val loss: %.2f  score: %s'\
            %(epoch, val_loss.average(),{key: score[key] for key in score}))
            model.save_networks(epoch)
        # if epoch<200:
        #     if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
        #         print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        #         model.save_networks('latest')
        #         model.save_networks(epoch)
        # else:
            
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))