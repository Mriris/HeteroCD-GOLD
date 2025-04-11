import argparse
import os
import time

import cv2
import seaborn as sns
import torch.nn.functional as FF
from scipy.spatial.distance import euclidean
from scipy.special import rel_entr
from scipy.stats import entropy, wasserstein_distance
from skimage import io
from torchvision import transforms

from models.DualEUNet import DualEUNet
from utils.util import get_confuse_matrix, cm2score

DATA_NAME = 'ST'
import torch
import numpy as np
import matplotlib.pyplot as plt


# 输入：两个BCHW张量
def visualize_features(tensor1, tensor2, output_filename, sample_size=10000):
    """
    可视化两组特征的直方图和KDE，并保存图像到指定文件。

    参数:
        tensor1 (torch.Tensor): 第一组特征，形状为 (B, C, H, W)。
        tensor2 (torch.Tensor): 第二组特征，形状为 (B, C, H, W)。
        output_filename (str): 图像保存的文件名（包含路径）。
        sample_size (int): 随机采样的大小，默认10000。
    """
    # 将特征展平成 1D 数组
    flattened_features1 = tensor1.flatten().cpu().numpy()
    flattened_features2 = tensor2.flatten().cpu().numpy()

    # 随机采样来加快计算
    sampled_features1 = np.random.choice(flattened_features1, size=sample_size, replace=False)
    sampled_features2 = np.random.choice(flattened_features2, size=sample_size, replace=False)

    # 创建一个2x2的绘图窗口
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # 绘制 tensor1 的直方图
    sns.histplot(sampled_features1, bins=50, color='blue', ax=axs[0, 0])
    axs[0, 0].set_title('Histogram of Features1')
    axs[0, 0].set_xlabel('Feature Values')
    axs[0, 0].set_ylabel('Count')

    # 绘制 tensor2 的直方图
    sns.histplot(sampled_features2, bins=50, color='red', ax=axs[0, 1])
    axs[0, 1].set_title('Histogram of Features2')
    axs[0, 1].set_xlabel('Feature Values')
    axs[0, 1].set_ylabel('Count')

    # 绘制 tensor1 的 KDE
    sns.kdeplot(sampled_features1, color='blue', ax=axs[1, 0])
    axs[1, 0].set_title('KDE of Features1')
    axs[1, 0].set_xlabel('Feature Values')
    axs[1, 0].set_ylabel('Density')

    # 绘制 tensor2 的 KDE
    sns.kdeplot(sampled_features2, color='red', ax=axs[1, 1])
    axs[1, 1].set_title('KDE of Features2')
    axs[1, 1].set_xlabel('Feature Values')
    axs[1, 1].set_ylabel('Density')

    # 调整图像布局
    plt.tight_layout()

    # 保存图像到文件
    plt.savefig(output_filename)
    plt.close()

# # 示例：创建两个随机的BCHW张量
# tensor1 = torch.rand(4, 64, 128, 128)
# tensor2 = torch.rand(4, 64, 128, 128)

# # 调用函数对比特征分布


def compute_feature_distances(tensor1, tensor2):
    # 检查输入的形状是否相同
    if tensor1.shape != tensor2.shape:
        raise ValueError("输入的两个tensor形状必须相同")

    # 将tensor展平为 [B, C * H * W] 的形状
    B = tensor1.size(0)  # batch size
    tensor1_flat = tensor1.view(B, -1).cpu().numpy()
    tensor2_flat = tensor2.view(B, -1).cpu().numpy()

    # 1. 欧氏距离 (Euclidean Distance)
    euclidean_distances = np.array([euclidean(t1, t2) for t1, t2 in zip(tensor1_flat, tensor2_flat)])
    euclidean_mean = np.mean(euclidean_distances)

    # 2. KL散度 (Kullback-Leibler Divergence)
    tensor1_prob = torch.nn.functional.softmax(torch.tensor(tensor1_flat), dim=-1).numpy()
    tensor2_prob = torch.nn.functional.softmax(torch.tensor(tensor2_flat), dim=-1).numpy()

    kl_divergence = np.array([np.sum(rel_entr(t1, t2)) for t1, t2 in zip(tensor1_prob, tensor2_prob)])
    kl_mean = np.mean(kl_divergence)

    # 3. JS散度 (Jensen-Shannon Divergence)
    def jensen_shannon_divergence(p, q):
        m = 0.5 * (p + q)
        return 0.5 * (entropy(p, m) + entropy(q, m))

    js_divergence = np.array([jensen_shannon_divergence(t1, t2) for t1, t2 in zip(tensor1_prob, tensor2_prob)])
    js_mean = np.mean(js_divergence)

    # 4. Wasserstein 距离 (Wasserstein Distance)
    wasserstein_distances = np.array([wasserstein_distance(t1, t2) for t1, t2 in zip(tensor1_flat, tensor2_flat)])
    wasserstein_mean = np.mean(wasserstein_distances)

    # 5. Hellinger 距离 (Hellinger Distance)
    def hellinger_distance(p, q):
        return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))

    hellinger_distances = np.array([hellinger_distance(t1, t2) for t1, t2 in zip(tensor1_prob, tensor2_prob)])
    hellinger_mean = np.mean(hellinger_distances)

    # 返回结果
    results = {
        "Euclidean Distance": euclidean_mean,
        "KL Divergence": kl_mean,
        "JS Divergence": js_mean,
        "Wasserstein Distance": wasserstein_mean,
        "Hellinger Distance": hellinger_mean
    }

    return results
class PredOptions:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        working_path = os.path.dirname(os.path.abspath(__file__))
        parser.add_argument('--pred_batch_size', type=int, default=1, help='prediction batch size')
        parser.add_argument('--test_dir', type=str, default='/data/jingwei/HeteCD/data/xiongan_data/val/', help='directory to test images')
        parser.add_argument('--pred_dir', type=str, default='/data/jingwei/HeteCD/HeteGAN/checkpoints/FreqHete_cosine_no_style_embed/res', help='directory to output masks')
        parser.add_argument('--chkpt_path', type=str, default='checkpoints/FreqHete_cosine_no_style_embed/be370_net_CD.pth', help='path to checkpoint')
        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        self.parser = parser
        return parser.parse_args()

    def parse(self):
        self.opt = self.gather_options()
        return self.opt


def main():
    begin_time = time.time()
    opt = PredOptions().parse()
    net = DualEUNet(3,3).cuda()
    checkpoint = torch.load(opt.chkpt_path, map_location='cuda:0')
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in checkpoint.items()}
    net.load_state_dict(new_state_dict)

    net.eval()



    predict(net, opt.test_dir, opt.pred_dir)
    time_use = time.time() - begin_time
    print('Total time: %.2fs' % time_use)


def load_images_from_folder(folder):
    images_A = []
    images_B = []
    labels = []
    filenames = sorted(os.listdir(os.path.join(folder, 'A')))
    for filename in filenames:
        img_A = io.imread(os.path.join(folder, 'A', filename))
        img_B = io.imread(os.path.join(folder, 'B', filename))
        label = io.imread(os.path.join(folder, 'label', filename))
        if img_A is not None and img_B is not None:
            images_A.append(img_A)
            images_B.append(img_B)
            labels.append(label)
    return images_A, images_B, labels, filenames


def predict(net, test_dir, pred_dir):
    images_A, images_B, labels, filenames = load_images_from_folder(test_dir)
    preds_all = []
    labels_all = []
    fenbuA1 = []
    fenbuA2 = []
    fenbuB1 = []
    fenbuB2 = []
    for img_A, img_B, label, filename in zip(images_A, images_B, labels, filenames):
        # img_A = torch.from_numpy(img_A).permute(2, 0, 1).unsqueeze(0).cuda().float()
        # img_B = torch.from_numpy(img_B).permute(2, 0, 1).unsqueeze(0).cuda().float()
        transform_list = [transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        img_transform = transforms.Compose(transform_list)
        img_A = img_transform(img_A)
        img_B = img_transform(img_B)
        with torch.no_grad():
            out_change,_x_sar,x_opt= net(img_A.unsqueeze(0).cuda().float(), img_B.unsqueeze(0).cuda().float())
            out_change = FF.interpolate(out_change, size=(512,512), mode='bilinear', align_corners=True)
            fenbuA1.append(_x_sar[-1])
            fenbuB1.append(x_opt[-1])
            preds = torch.argmax(out_change, dim=1)
            pred_numpy = preds.cpu().numpy()
            # pred_numpy = preds.cpu().numpy()
            labels_numpy = label[np.newaxis, ...]
            preds_all.append(pred_numpy)
            labels_all.append(labels_numpy)


        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
        save_path = os.path.join(pred_dir, filename)
        print(filename)
        preds = preds.cpu().numpy().astype(np.uint8) * 255
        cv2.imwrite(save_path, preds[0])
    fenbuA1 = torch.cat(fenbuA1, dim=0)
    fenbuB1 = torch.cat(fenbuB1, dim=0)
    # featureA = torch.cat([fenbuA1, fenbuA2], dim=0)
    # featureB = torch.cat([fenbuB1, fenbuB2], dim=0)
    print(fenbuA1.shape)
    fenbu1_dis = compute_feature_distances(fenbuA1, fenbuB1)   
    print(fenbu1_dis)
    preds_all = np.concatenate(preds_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)//255
    print(preds_all.max())
    print(labels_all.max())
    print(preds_all.shape)
    print(labels_all.shape)
    
    hist = get_confuse_matrix(2,labels_all,preds_all)
    score = cm2score(hist)
    print(score)
if __name__ == '__main__':
    main()