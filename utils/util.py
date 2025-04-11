"""本模块包含一些简单的辅助函数"""
from __future__ import print_function

import math
import os
import random

import numpy as np
import torch
from PIL import Image
from scipy import stats

from utils import eval_segm as seg_acc


def get_square(img, pos):
    """从ndarray形状(H, W, C)中提取左侧或右侧正方形"""
    h = img.shape[0]
    if pos == 0:
        return img[:, :h]
    else:
        return img[:, -h:]


def split_img_into_squares(img):
    return get_square(img, 0), get_square(img, 1)


def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])


def resize_and_crop(pilimg, scale=0.5, final_height=None):
    w = pilimg.size[0]
    h = pilimg.size[1]
    newW = int(w * scale)
    newH = int(h * scale)

    if not final_height:
        diff = 0
    else:
        diff = newH - final_height

    img = pilimg.resize((newW, newH))
    img = img.crop((0, diff // 2, newW, newH - diff // 2))
    return np.array(img, dtype=np.float32)


def batch(iterable, batch_size):
    """按批次生成列表"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b


def seprate_batch(dataset, batch_size):
    """按批次分离数据集"""
    num_batch = len(dataset) // batch_size + 1
    batch_len = batch_size
    batches = []
    for i in range(num_batch):
        batches.append([dataset[j] for j in range(batch_len)])
        if i + 2 == num_batch: batch_len = len(dataset) - (num_batch - 1) * batch_size
    return batches


def split_train_val(dataset, val_percent=0.05):
    """将数据集分割为训练集和验证集"""
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.shuffle(dataset)
    return {'train': dataset[:-n], 'val': dataset[-n:]}


def normalize(x):
    """将图像数据归一化到0-1范围"""
    return x / 255


def merge_masks(img1, img2, full_w):
    """合并两个掩码图像"""
    h = img1.shape[0]

    new = np.zeros((h, full_w), np.float32)
    new[:, :full_w // 2 + 1] = img1[:, :full_w // 2 + 1]
    new[:, full_w // 2 + 1:] = img2[:, -(full_w // 2 - 1):]

    return new


def rle_encode(mask_image):
    """使用游程编码(RLE)压缩掩码图像"""
    pixels = mask_image.flatten()
    # 我们通过显式地将开始或结束处的'1'（在原始图像的角落）设置为'0'来避免问题。
    # 对于准确的掩码，我们不期望这些为非零，所以这不应该影响分数。
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


class AverageMeter(object):
    """计算并存储平均值和当前值"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, count, weight):
        self.val = val
        self.avg = val
        self.count = count
        self.sum = val * weight
        self.initialized = True

    def update(self, val, count=1, weight=1):
        if not self.initialized:
            self.initialize(val, count, weight)
        else:
            self.add(val, count, weight)

    def add(self, val, count, weight):
        self.val = val
        self.count += count
        self.sum += val * weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def ImageValStretch2D(img):
    """将图像值拉伸到0-255范围"""
    img = img * 255
    return img.astype(int)


def ConfMap(output, pred):
    """计算置信度图"""
    n, h, w = output.shape
    conf = np.zeros(pred.shape, float)
    for h_idx in range(h):
        for w_idx in range(w):
            n_idx = int(pred[h_idx, w_idx])
            sum = 0
            for i in range(n):
                val = output[i, h_idx, w_idx]
                if val > 0: sum += val
            conf[h_idx, w_idx] = output[n_idx, h_idx, w_idx] / sum
            if conf[h_idx, w_idx] < 0: conf[h_idx, w_idx] = 0
    return conf


def get_confuse_matrix(num_classes, label_gts, label_preds):
    """计算混淆矩阵
    
    参数:
        num_classes (int): 类别数量
        label_gts (list): 真实标签列表
        label_preds (list): 预测标签列表
        
    返回:
        confusion_matrix (np.ndarray): 混淆矩阵
    """

    def __fast_hist(label_gt, label_pred):
        """
        收集混淆矩阵的值
        参考: https://en.wikipedia.org/wiki/Confusion_matrix
        
        参数:
            label_gt (np.array): 真实标签
            label_pred (np.array): 预测标签
            
        返回:
            hist (np.ndarray): 用于混淆矩阵的值
        """
        mask = (label_gt >= 0) & (label_gt < num_classes)
        hist = np.bincount(num_classes * label_gt[mask].astype(int) + label_pred[mask],
                           minlength=num_classes ** 2).reshape(num_classes, num_classes)
        return hist

    confusion_matrix = np.zeros((num_classes, num_classes))
    for lt, lp in zip(label_gts, label_preds):
        confusion_matrix += __fast_hist(lt.flatten(), lp.flatten())
    return confusion_matrix


def cm2F1(confusion_matrix):
    """从混淆矩阵计算F1分数"""
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)

    # 1. 准确率 & 类别准确率
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    # 召回率
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)

    # 精确率
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1分数
    F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)
    return mean_F1


def cm2score(confusion_matrix):
    """从混淆矩阵计算各种评估指标"""
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)

    # 1. 准确率 & 类别准确率
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    # 召回率
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)

    # 精确率
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1分数
    F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)

    # 2. 频率加权准确率 & 平均IoU
    iu = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)
    mean_iu = np.nanmean(iu)

    freq = sum_a1 / (hist.sum() + np.finfo(np.float32).eps)
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    # 创建输出字典
    cls_iou = dict(zip(['iou_' + str(i) for i in range(n_class)], iu))

    cls_precision = dict(zip(['precision_' + str(i) for i in range(n_class)], precision))
    cls_recall = dict(zip(['recall_' + str(i) for i in range(n_class)], recall))
    cls_F1 = dict(zip(['F1_' + str(i) for i in range(n_class)], F1))

    score_dict = {'acc': acc, 'miou': mean_iu, 'mf1': mean_F1}
    score_dict.update(cls_iou)
    score_dict.update(cls_F1)
    score_dict.update(cls_precision)
    score_dict.update(cls_recall)
    return score_dict


def accuracy(pred, label):
    # print("pred:",pred.max(), pred.min())
    # print("label:",label.max(), label.min())
    valid = (label >= 0)

    acc_sum = (valid * (pred == label)).sum()
    valid_sum = valid.sum()
    # print("acc_sum:", acc_sum)
    # print("valid_sum:", valid_sum)
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def get_hist(image, label, num_class):
    hist = np.zeros((num_class, num_class))
    hist += fast_hist(image.flatten(), label.flatten(), num_class)
    return hist


def cal_kappa(hist):
    if hist.sum() == 0:
        po = 0
        pe = 1
        kappa = 0
    else:
        po = np.diag(hist).sum() / hist.sum()
        pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
        if pe == 1:
            kappa = 0
        else:
            kappa = (po - pe) / (1 - pe)
    return kappa


def SCDD_eval_all(preds, labels, num_class):
    hist = np.zeros((num_class, num_class))
    for pred, label in zip(preds, labels):
        infer_array = np.array(pred)
        unique_set = set(np.unique(infer_array))
        assert unique_set.issubset(set([0, 1, 2, 3, 4, 5, 6])), "unrecognized label number"
        label_array = np.array(label)
        assert infer_array.shape == label_array.shape, "The size of prediction and target must be the same"
        hist += get_hist(infer_array, label_array, num_class)

    hist_fg = hist[1:, 1:]
    c2hist = np.zeros((2, 2))
    c2hist[0][0] = hist[0][0]
    c2hist[0][1] = hist.sum(1)[0] - hist[0][0]
    c2hist[1][0] = hist.sum(0)[0] - hist[0][0]
    c2hist[1][1] = hist_fg.sum()
    hist_n0 = hist.copy()
    hist_n0[0][0] = 0
    kappa_n0 = cal_kappa(hist_n0)
    iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
    IoU_fg = iu[1]
    IoU_mean = (iu[0] + iu[1]) / 2
    Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e

    pixel_sum = hist.sum()
    change_pred_sum = pixel_sum - hist.sum(1)[0].sum()
    change_label_sum = pixel_sum - hist.sum(0)[0].sum()
    change_ratio = change_label_sum / pixel_sum
    SC_TP = np.diag(hist[1:, 1:]).sum()
    SC_Precision = SC_TP / change_pred_sum
    SC_Recall = SC_TP / change_label_sum
    Fscd = stats.hmean([SC_Precision, SC_Recall])
    return Fscd, IoU_mean, Sek


def SCDD_eval(pred, label, num_class):
    infer_array = np.array(pred)
    unique_set = set(np.unique(infer_array))
    assert unique_set.issubset(set([0, 1, 2, 3, 4, 5, 6])), "unrecognized label number"
    label_array = np.array(label)
    assert infer_array.shape == label_array.shape, "The size of prediction and target must be the same"
    hist = get_hist(infer_array, label_array, num_class)
    hist_fg = hist[1:, 1:]
    c2hist = np.zeros((2, 2))
    c2hist[0][0] = hist[0][0]
    c2hist[0][1] = hist.sum(1)[0] - hist[0][0]
    c2hist[1][0] = hist.sum(0)[0] - hist[0][0]
    c2hist[1][1] = hist_fg.sum()
    hist_n0 = hist.copy()
    hist_n0[0][0] = 0
    kappa_n0 = cal_kappa(hist_n0)
    iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
    IoU_fg = iu[1]
    IoU_mean = (iu[0] + iu[1]) / 2
    Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e

    pixel_sum = hist.sum()
    change_pred_sum = pixel_sum - hist.sum(1)[0].sum()
    change_label_sum = pixel_sum - hist.sum(0)[0].sum()
    change_ratio = change_label_sum / pixel_sum
    SC_TP = np.diag(hist[1:, 1:]).sum()
    SC_Precision = SC_TP / change_pred_sum
    SC_Recall = SC_TP / change_label_sum
    Fscd = stats.hmean([SC_Precision, SC_Recall])
    return Fscd, IoU_mean, Sek


def FWIoU(pred, label, bn_mode=False, ignore_zero=False):
    if bn_mode:
        pred = (pred >= 0.5)
        label = (label >= 0.5)
    elif ignore_zero:
        pred = pred - 1
        label = label - 1
    FWIoU = seg_acc.frequency_weighted_IU(pred, label)
    return FWIoU


def binary_accuracy(pred, label):
    valid = (label < 2)
    acc_sum = (valid * (pred == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc


def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    imPred += 1
    imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass + 1))
    # print(area_intersection)

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass + 1))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass + 1))
    area_union = area_pred + area_lab - area_intersection
    # print(area_pred)
    # print(area_lab)

    return area_intersection, area_union


def CaclTP(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    imPred += 1
    imLab += 1
    # # Remove classes from unlabeled pixels in gt image.
    # # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    TP = imPred * (imPred == imLab)
    (TP_hist, _) = np.histogram(
        TP, bins=numClass, range=(1, numClass + 1))
    # print(TP.shape)
    # print(TP_hist)

    # Compute area union:
    (pred_hist, _) = np.histogram(imPred, bins=numClass, range=(1, numClass + 1))
    (lab_hist, _) = np.histogram(imLab, bins=numClass, range=(1, numClass + 1))
    # print(pred_hist)
    # print(lab_hist)
    # precision = TP_hist / (lab_hist + 1e-10) + 1e-10
    # recall = TP_hist / (pred_hist + 1e-10) + 1e-10
    # # print(precision)
    # # print(recall)
    # F1 = [stats.hmean([pre, rec]) for pre, rec in zip(precision, recall)]
    # print(F1)

    # print(area_pred)
    # print(area_lab)

    return TP_hist, pred_hist, lab_hist


def tensor2im(input_image, imtype=np.uint8):
    """将张量数组转换为numpy图像数组
    
    参数:
        input_image (torch.Tensor): 输入图像张量
        imtype (type): 期望的输出图像类型
        
    返回:
        numpy数组格式的图像
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))

        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name + ": 平均绝对梯度值: " + str(mean))


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
