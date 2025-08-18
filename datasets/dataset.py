import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from PIL import ImageFilter
from scipy.ndimage import uniform_filter
# from osgeo import gdal_array
from skimage import io, img_as_float
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional as F

MEAN_A = np.array([113.40, 114.08, 116.45])
STD_A = np.array([48.30, 46.27, 48.14])
MEAN_B = np.array([111.07, 111.07, 111.07])
STD_B = np.array([49.41, 49.41, 49.41])


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    new_h = new_w = opt.load_size
    new_w = opt.load_size
    new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.load_size))
    y = random.randint(0, np.maximum(0, new_h - opt.load_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}


class CDDataAugmentation:

    def __init__(
            self,
            img_size,
            with_random_hflip=False,
            with_random_vflip=False,
            with_random_rot=False,
            with_random_crop=False,
            with_scale_random_crop=False,
            with_random_blur=False,
            random_color_tf=False,
            rotate_prob=0.0,
            rotate_degree=0,
            hflip_prob=0.0,
            vflip_prob=0.0,
            exchange_time_prob=0.0,
            use_photometric=True,
            brightness_delta=10,
            contrast_range=(0.8, 1.2),
            saturation_range=(0.8, 1.2),
            hue_delta=10,
            cat_max_ratio=0.75,
            ignore_index=255
    ):
        self.img_size = img_size
        if self.img_size is None:
            self.img_size_dynamic = True
        else:
            self.img_size_dynamic = False
        self.hflip_prob = hflip_prob if hflip_prob is not None else (0.5 if with_random_hflip else 0.0)
        self.vflip_prob = vflip_prob if vflip_prob is not None else (0.5 if with_random_vflip else 0.0)
        self.rotate_prob = rotate_prob if rotate_prob is not None else (0.5 if with_random_rot else 0.0)
        self.rotate_degree = rotate_degree
        self.with_random_crop = with_random_crop
        self.with_scale_random_crop = with_scale_random_crop
        self.with_random_blur = with_random_blur
        # photometric 设置
        self.use_photometric = use_photometric or random_color_tf
        self.brightness_delta = brightness_delta
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_delta = hue_delta
        # 其他
        self.exchange_time_prob = exchange_time_prob
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def transform(self, imgs, labels, to_tensor=True):
        """
        :param imgs: [ndarray,]
        :param labels: [ndarray,]
        :return: [ndarray,],[ndarray,]
        """
        # 调整图像大小并转换为PIL
        imgs = [F.to_pil_image(img) for img in imgs]
        if self.img_size is None:
            self.img_size = None

        if not self.img_size_dynamic:
            if imgs[0].size != (self.img_size, self.img_size):
                imgs = [F.resize(img, [self.img_size, self.img_size], interpolation=3)
                        for img in imgs]
        else:
            self.img_size = imgs[0].size[0]

        labels = [F.to_pil_image(img) for img in labels]
        if len(labels) != 0:
            if labels[0].size != (self.img_size, self.img_size):
                labels = [F.resize(img, [self.img_size, self.img_size], interpolation=0)
                          for img in labels]

        # 1) 时间交换（A/B交换）
        if self.exchange_time_prob > 0 and len(imgs) >= 2:
            if random.random() < self.exchange_time_prob:
                imgs[0], imgs[1] = imgs[1], imgs[0]

        # 2) 随机旋转（对齐OpenCD: prob, ±degree）
        if self.rotate_prob > 0 and self.rotate_degree > 0 and random.random() < self.rotate_prob:
            angle = random.uniform(-float(self.rotate_degree), float(self.rotate_degree))
            imgs = [img.rotate(angle, resample=Image.BILINEAR, expand=False) for img in imgs]
            labels = [img.rotate(angle, resample=Image.NEAREST, expand=False) for img in labels]

        # 3) 随机裁剪（带cat_max_ratio重试机制，最多10次），若无标签则跳过占比检查
        if self.with_random_crop:
            imgsize = imgs[0].size
            selected_box = None
            if len(labels) == 0 or self.cat_max_ratio >= 1.0:
                selected_box = get_random_crop_box(imgsize=imgsize, cropsize=self.img_size)
            else:
                for _ in range(10):
                    box = get_random_crop_box(imgsize=imgsize, cropsize=self.img_size)
                    lbl_crop = pil_crop(labels[0], box, cropsize=self.img_size, default_value=self.ignore_index)
                    lbl_np = np.array(lbl_crop)
                    labels_unique, cnt = np.unique(lbl_np, return_counts=True)
                    mask = labels_unique != self.ignore_index
                    cnt = cnt[mask]
                    if cnt.size == 0:
                        selected_box = box
                        break
                    if (np.max(cnt) / np.sum(cnt)) < self.cat_max_ratio:
                        selected_box = box
                        break
                # 若十次都未满足，占比条件，则使用最后一个box
                if selected_box is None:
                    selected_box = get_random_crop_box(imgsize=imgsize, cropsize=self.img_size)

            imgs = [pil_crop(img, selected_box, cropsize=self.img_size, default_value=0) for img in imgs]
            labels = [pil_crop(img, selected_box, cropsize=self.img_size, default_value=self.ignore_index) for img in labels]

        # 4) 随机翻转（水平/垂直，按概率）
        if self.hflip_prob > 0 and random.random() < self.hflip_prob:
            imgs = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in imgs]
            labels = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in labels]
        if self.vflip_prob > 0 and random.random() < self.vflip_prob:
            imgs = [img.transpose(Image.FLIP_TOP_BOTTOM) for img in imgs]
            labels = [img.transpose(Image.FLIP_TOP_BOTTOM) for img in labels]

        # 5) 可选：尺度随机裁剪（保持原有功能）
        if self.with_scale_random_crop:
            # 缩放
            scale_range = [1, 1.2]
            target_scale = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
            imgs = [pil_rescale(img, target_scale, order=3) for img in imgs]
            labels = [pil_rescale(img, target_scale, order=0) for img in labels]
            # 裁剪
            imgsize = imgs[0].size  # h, w
            box = get_random_crop_box(imgsize=imgsize, cropsize=self.img_size)
            imgs = [pil_crop(img, box, cropsize=self.img_size, default_value=0)
                    for img in imgs]
            labels = [pil_crop(img, box, cropsize=self.img_size, default_value=self.ignore_index)
                      for img in labels]

        # 6) 可选：随机模糊（保持原有功能）
        if self.with_random_blur and random.random() > 0:
            radius = random.random()
            imgs = [img.filter(ImageFilter.GaussianBlur(radius=radius))
                    for img in imgs]

        # 7) 光照与颜色扰动（对齐OpenCD PhotoMetricDistortion，使用ColorJitter近似）
        if self.use_photometric:
            # brightness_delta是加性扰动，这里用乘性近似到 [1-d/255, 1+d/255]
            bd = max(float(self.brightness_delta), 0.0)
            bmin = max(0.0, 1.0 - bd / 255.0)
            bmax = 1.0 + bd / 255.0
            cmin, cmax = float(self.contrast_range[0]), float(self.contrast_range[1])
            smin, smax = float(self.saturation_range[0]), float(self.saturation_range[1])
            # hue_delta按度近似到[0,0.5]范围
            h = max(float(self.hue_delta), 0.0) / 180.0
            jitter = transforms.ColorJitter(
                brightness=(bmin, bmax),
                contrast=(cmin, cmax),
                saturation=(smin, smax),
                hue=h
            )
            imgs = [jitter(img) for img in imgs]

        return imgs, labels


def pil_crop(image, box, cropsize, default_value):
    assert isinstance(image, Image.Image)
    img = np.array(image)

    if len(img.shape) == 3:
        cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype) * default_value
    else:
        cont = np.ones((cropsize, cropsize), img.dtype) * default_value
    cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]

    return Image.fromarray(cont)


def get_random_crop_box(imgsize, cropsize):
    h, w = imgsize
    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0

    return cont_top, cont_top + ch, cont_left, cont_left + cw, img_top, img_top + ch, img_left, img_left + cw


def pil_rescale(img, scale, order):
    assert isinstance(img, Image.Image)
    height, width = img.size
    target_size = (int(np.round(height * scale)), int(np.round(width * scale)))
    return pil_resize(img, target_size, order)


def pil_resize(img, size, order):
    assert isinstance(img, Image.Image)
    if size[0] == img.size[0] and size[1] == img.size[1]:
        return img
    if order == 3:
        resample = Image.BICUBIC
    elif order == 0:
        resample = Image.NEAREST
    return img.resize(size[::-1], resample)


def showIMG(img):
    plt.imshow(img)
    plt.show()
    return 0


def lee_filter(img, size):
    img = img_as_float(img)  # 转换图像到浮点型，范围在0到1
    mean_img = uniform_filter(img, size)  # 计算局部均值
    mean_sqr_img = uniform_filter(img ** 2, size)  # 计算局部平方均值
    var_img = mean_sqr_img - mean_img ** 2  # 计算局部方差
    noise = np.mean(var_img)
    coeff = var_img / (var_img + noise)
    result_img = mean_img + coeff * (img - mean_img)
    return result_img


def normalize_image(im, time='A'):
    assert time in ['A', 'B']
    im = im / 255
    return im


def read_RSimages(mode, rescale=False, root=None, opt=None, load_t2_opt=False):
    if root is None and opt is not None:
        root = opt.dataroot  # 从 opt 中获取 dataroot
    if root is None:
        raise ValueError("root 参数未指定，请通过 opt.dataroot 或直接设置 root 提供数据路径")

    # 如果mode是test，则使用val文件夹
    if mode == 'test':
        mode = 'val'
        print("使用验证集(val)数据作为测试集(test)数据")

    img_A_dir = os.path.join(root, mode, 'A')
    img_B_dir = os.path.join(root, mode, 'B')
    label_A_dir = os.path.join(root, mode, 'label')
    
    # 如果需要加载时间点2的光学图像，也读取C文件夹
    if load_t2_opt:
        img_C_dir = os.path.join(root, mode, 'C')
        if not os.path.exists(img_C_dir):
            print(f"警告：未找到时间点2的光学图像文件夹({img_C_dir})，将尝试使用A文件夹作为替代")
            img_C_dir = img_A_dir  # 如果C文件夹不存在，使用A文件夹作为替代
    
    data_list = os.listdir(img_A_dir)
    imgs_list_A, imgs_list_B, labels_A = [], [], []
    imgs_list_C = [] if load_t2_opt else None
    
    count = 0
    for it in data_list:
        img_A_path = os.path.join(img_A_dir, it)
        img_B_path = os.path.join(img_B_dir, it)
        label_A_path = os.path.join(label_A_dir, it)
        
        imgs_list_A.append(img_A_path)
        imgs_list_B.append(img_B_path)
        
        # 如果需要加载时间点2的光学图像
        if load_t2_opt:
            img_C_path = os.path.join(img_C_dir, it)
            imgs_list_C.append(img_C_path)
        
        label_A = io.imread(label_A_path)
        labels_A.append(label_A)
        count += 1
        if not count % 500: print('已加载 %d/%d 张图像' % (count, len(data_list)))

    # 转换输出为中文
    mode_name = {
        'train': '训练',
        'val': '验证',
        'test': '测试'
    }.get(mode, mode)

    print('已加载 ' + str(len(imgs_list_A)) + ' 张' + mode_name + '图像')

    if load_t2_opt:
        return imgs_list_A, imgs_list_B, labels_A, imgs_list_C
    else:
        return imgs_list_A, imgs_list_B, labels_A


def __transforms2pil_resize(method):
    mapper = {transforms.InterpolationMode.BILINEAR: Image.BILINEAR,
              transforms.InterpolationMode.BICUBIC: Image.BICUBIC,
              transforms.InterpolationMode.NEAREST: Image.NEAREST,
              transforms.InterpolationMode.LANCZOS: Image.LANCZOS, }
    return mapper[method]


def __make_power_2(img, base, method=transforms.InterpolationMode.BICUBIC):
    method = __transforms2pil_resize(method)
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __print_size_warning(ow, oh, w, h):
    """打印图像大小警告信息（仅打印一次）"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("图像尺寸需要是4的倍数。"
              "加载的图像尺寸为 (%d, %d)，已调整为"
              "(%d, %d)。此调整将应用于"
              "所有尺寸不是4的倍数的图像" % (ow, oh, w, h))
        __print_size_warning.has_printed = True


def __scale_width(img, target_size, crop_size, method=transforms.InterpolationMode.BICUBIC):
    method = __transforms2pil_resize(method)
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if ow > tw or oh > th:
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def get_transform(opt, params=None, grayscale=False, method=transforms.InterpolationMode.NEAREST, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    osize = [opt.load_size, opt.load_size]
    transform_list.append(transforms.Resize(osize, method))

    transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))

    if params is None:
        transform_list.append(transforms.RandomCrop(opt.crop_size))
    else:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    transform_list.append(transforms.RandomHorizontalFlip())

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


class Data(data.Dataset):
    def __init__(self, mode, random_flip=False, root=None, opt=None, load_t2_opt=False):
        if root is None and opt is not None:
            root = opt.dataroot  # 从 opt 中获取 dataroot
        if root is None:
            raise ValueError("root 参数未指定，请通过 opt.dataroot 或直接设置 root 提供数据路径")
        self.random_flip = random_flip
        self.load_t2_opt = load_t2_opt
        
        # 根据需要是否加载时间点2的光学图像
        if load_t2_opt:
            self.imgs_list_A, self.imgs_list_B, self.labels, self.imgs_list_C = read_RSimages(
                mode, root=root, opt=opt, load_t2_opt=True
            )
        else:
            self.imgs_list_A, self.imgs_list_B, self.labels = read_RSimages(
                mode, root=root, opt=opt, load_t2_opt=False
            )
            
        self.mode = mode
        self.opt = opt
        self.augm = CDDataAugmentation(
            img_size=opt.crop_size if opt is not None else 512,
            with_random_crop=True,
            with_scale_random_crop=(opt.aug_use_scale_random_crop if (opt is not None and hasattr(opt, 'aug_use_scale_random_crop')) else True),
            with_random_blur=(opt.aug_use_random_blur if (opt is not None and hasattr(opt, 'aug_use_random_blur')) else True),
            # 开关来自TrainOptions（OpenCD风格）
            rotate_prob=(opt.aug_rotate_prob if (opt is not None and hasattr(opt, 'aug_rotate_prob')) else 0.0),
            rotate_degree=(opt.aug_rotate_degree if (opt is not None and hasattr(opt, 'aug_rotate_degree')) else 0),
            hflip_prob=(opt.aug_hflip_prob if (opt is not None and hasattr(opt, 'aug_hflip_prob')) else 0.0),
            vflip_prob=(opt.aug_vflip_prob if (opt is not None and hasattr(opt, 'aug_vflip_prob')) else 0.0),
            exchange_time_prob=(opt.aug_exchange_time_prob if (opt is not None and hasattr(opt, 'aug_exchange_time_prob')) else 0.0),
            use_photometric=(opt.aug_use_photometric if (opt is not None and hasattr(opt, 'aug_use_photometric')) else True),
            brightness_delta=(opt.aug_brightness_delta if (opt is not None and hasattr(opt, 'aug_brightness_delta')) else 10),
            contrast_range=(tuple(opt.aug_contrast_range) if (opt is not None and hasattr(opt, 'aug_contrast_range')) else (0.8, 1.2)),
            saturation_range=(tuple(opt.aug_saturation_range) if (opt is not None and hasattr(opt, 'aug_saturation_range')) else (0.8, 1.2)),
            hue_delta=(opt.aug_hue_delta if (opt is not None and hasattr(opt, 'aug_hue_delta')) else 10),
            cat_max_ratio=(opt.aug_cat_max_ratio if (opt is not None and hasattr(opt, 'aug_cat_max_ratio')) else 0.75)
        )

    def get_mask_name(self, idx):
        mask_name = os.path.split(self.imgs_list_A[idx])[-1]
        return mask_name

    def __getitem__(self, idx):
        img_A = io.imread(self.imgs_list_A[idx])
        base_name = os.path.basename(self.imgs_list_A[idx])
        name = os.path.splitext(base_name)[0] + '.png'
        img_B = io.imread(self.imgs_list_B[idx])
        
        # 如果需要加载时间点2的光学图像
        if self.load_t2_opt:
            img_C = io.imread(self.imgs_list_C[idx])
        
        transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.img_transform = transforms.Compose(transform_list)
        label = self.labels[idx] // 255
        
        # 数据增强处理，确保所有图像采用相同的转换
        if self.mode == "train" and (self.opt is None or getattr(self.opt, 'aug_in_train', True)):
            if self.load_t2_opt:
                [img_A, img_B, img_C], [label] = self.augm.transform([img_A, img_B, img_C], [label])
            else:
                [img_A, img_B], [label] = self.augm.transform([img_A, img_B], [label])
        
        # 转换为张量
        img_A = self.img_transform(img_A)
        img_B = self.img_transform(img_B)
        if self.load_t2_opt:
            img_C = self.img_transform(img_C)
        
        label = torch.from_numpy(np.array(label, np.uint8)).unsqueeze(dim=0)

        if self.load_t2_opt:
            return img_A, img_B, label.squeeze(), name, img_C
        else:
            return img_A, img_B, label.squeeze(), name

    def __len__(self):
        return len(self.imgs_list_A)
