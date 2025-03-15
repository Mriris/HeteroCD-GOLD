import os
import numpy as np
import torch
from skimage import io
from torch.utils import data
import matplotlib.pyplot as plt
from skimage.transform import rescale
from torchvision.transforms import functional as F
import random
from PIL import Image
from PIL import ImageFilter
from torchvision import transforms
# from osgeo import gdal_array
import cv2
from skimage import io, img_as_float
from scipy.ndimage import uniform_filter

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
            random_color_tf=False
    ):
        self.img_size = img_size
        if self.img_size is None:
            self.img_size_dynamic = True
        else:
            self.img_size_dynamic = False
        self.with_random_hflip = with_random_hflip
        self.with_random_vflip = with_random_vflip
        self.with_random_rot = with_random_rot
        self.with_random_crop = with_random_crop
        self.with_scale_random_crop = with_scale_random_crop
        self.with_random_blur = with_random_blur
        self.random_color_tf = random_color_tf

    def transform(self, imgs, labels, to_tensor=True):
        """
        :param imgs: [ndarray,]
        :param labels: [ndarray,]
        :return: [ndarray,],[ndarray,]
        """
        # resize image and covert to tensor
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

        random_base = 0.5
        # if self.with_random_hflip and random.random() > 0.5:
        #     imgs = [F.hflip(img) for img in imgs]
        #     labels = [F.hflip(img) for img in labels]

        # if self.with_random_vflip and random.random() > 0.5:
        #     imgs = [F.vflip(img) for img in imgs]
        #     labels = [F.vflip(img) for img in labels]

        # if self.with_random_rot and random.random() > random_base:
        #     angles = [90, 180, 270]
        #     index = random.randint(0, 2)
        #     angle = angles[index]
        #     imgs = [F.rotate(img, angle) for img in imgs]
        #     labels = [F.rotate(img, angle) for img in labels]

        if self.with_random_crop and random.random() > 0:
            i, j, h, w = transforms.RandomResizedCrop(size=self.img_size). \
                get_params(img=imgs[0], scale=(0.8, 1.2), ratio=(1, 1))

            imgs = [F.resized_crop(img, i, j, h, w,
                                   size=(self.img_size, self.img_size),
                                   interpolation=Image.CUBIC)
                    for img in imgs]

            labels = [F.resized_crop(img, i, j, h, w,
                                     size=(self.img_size, self.img_size),
                                     interpolation=Image.NEAREST)
                      for img in labels]

        if self.with_scale_random_crop:
            # rescale
            scale_range = [1, 1.2]
            target_scale = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])

            imgs = [pil_rescale(img, target_scale, order=3) for img in imgs]
            labels = [pil_rescale(img, target_scale, order=0) for img in labels]
            # crop
            imgsize = imgs[0].size  # h, w
            box = get_random_crop_box(imgsize=imgsize, cropsize=self.img_size)
            imgs = [pil_crop(img, box, cropsize=self.img_size, default_value=0)
                    for img in imgs]
            labels = [pil_crop(img, box, cropsize=self.img_size, default_value=255)
                      for img in labels]

        if self.with_random_blur and random.random() > 0:
            radius = random.random()
            imgs = [img.filter(ImageFilter.GaussianBlur(radius=radius))
                    for img in imgs]

        if self.random_color_tf:
            color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)
            imgs_tf = []
            for img in imgs:
                tf = transforms.ColorJitter(
                    color_jitter.brightness,
                    color_jitter.contrast,
                    color_jitter.saturation,
                    color_jitter.hue)
                imgs_tf.append(tf(img))
            imgs = imgs_tf

        # if to_tensor:
        #     # to tensor
        #     imgs = [TF.to_tensor(img) for img in imgs]
        # labels = [torch.from_numpy(np.array(img, np.uint8)).unsqueeze(dim=0)
        #         for img in labels]

        #     imgs = [TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        #             for img in imgs]
        # img1 = np.array(imgs[0])
        # imgs = [F.to_tensor(img) for img in imgs]
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


def read_RSimages(mode, rescale=False, root=None):
    # assert mode in ['train', 'val', 'train_unchange']
    img_A_dir = os.path.join(root, mode, 'A')
    img_B_dir = os.path.join(root, mode, 'B')
    label_A_dir = os.path.join(root, mode, 'label')
    data_list = os.listdir(img_A_dir)
    imgs_list_A, imgs_list_B, labels_A = [], [], []
    count = 0
    for it in data_list:
        # print(it)
        # if (it[-4:]=='.tif'):
        img_A_path = os.path.join(img_A_dir, it)
        img_B_path = os.path.join(img_B_dir, it)
        label_A_path = os.path.join(label_A_dir, it)
        imgs_list_A.append(img_A_path)
        imgs_list_B.append(img_B_path)
        label_A = io.imread(label_A_path)
        labels_A.append(label_A)
        count += 1
        if not count % 500: print('%d/%d images loaded.' % (count, len(data_list)))

    print(str(len(imgs_list_A)) + ' ' + mode + ' images' + ' loaded.')

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
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
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
    if (ow > tw or oh > th):
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


# class Data(data.Dataset):
#     def __init__(self, mode, img_transform, root = '/data/jingwei/HeteCD/data/xiongan_data'):
#         self.imgs_list_A, self.imgs_list_B, self.labels = read_RSimages(mode, root=root)
#         self.mode = mode
#         self.img_transform = img_transform
#         self.augm = CDDataAugmentation(
#                     img_size=512,
#                     with_random_hflip=True,
#                     with_random_vflip=True,
#                     with_scale_random_crop=True,
#                     with_random_blur=False,
#                     random_color_tf=True
#                 )
#     def get_mask_name(self, idx):
#         mask_name = os.path.split(self.imgs_list_A[idx])[-1]
#         return mask_name

#     def __getitem__(self, idx):
#         img_A = Image.open(self.imgs_list_A[idx]).convert('RGB')
#         # img_A = img_A[:, :, (2, 1, 0)]
#         name = self.imgs_list_A[idx].split('/')[-1]
#         img_B = Image.open(self.imgs_list_B[idx]).convert('RGB')
#         # img_B = img_B[:, :, (2, 1, 0)]
#         label= self.labels[idx]//255
#         # label = Image.fromarray(label)
#         # if self.mode=="train":
#         #     [img_A, img_B], [label] = self.augm.transform([img_A, img_B], [label])
#         # transform_list = [transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
#         # IMG_transform = transforms.Compose(transform_list)
#         # img_A = IMG_transform(img_A)
#         # img_B = IMG_transform(img_B)

#         # if self.mode!="train":
#         transform_list = [transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
#         self.img_transform = transforms.Compose(transform_list)
#         img_A = self.img_transform(img_A)
#         img_B = self.img_transform(img_B)

#         # label = self.img_transform(label)
#         # img_A = F.to_tensor(img_A)*2.0-1.0
#         # img_B = F.to_tensor(img_B)*2.0-1.0
#         label = torch.from_numpy(np.array(label, np.uint8)).unsqueeze(dim=0)
#         return img_A, img_B, label.squeeze(), name

#     def __len__(self):
#         return len(self.imgs_list_A)

class Data(data.Dataset):
    def __init__(self, mode, random_flip=False, root=None):
        self.random_flip = random_flip
        self.imgs_list_A, self.imgs_list_B, self.labels = read_RSimages(mode, root)
        self.mode = mode
        self.augm = CDDataAugmentation(
            img_size=512,
            with_random_hflip=True,
            with_random_vflip=True,
            with_scale_random_crop=True,
            with_random_blur=True,
            random_color_tf=False
        )

    def get_mask_name(self, idx):
        mask_name = os.path.split(self.imgs_list_A[idx])[-1]
        return mask_name

    def __getitem__(self, idx):
        img_A = io.imread(self.imgs_list_A[idx])
        name = self.imgs_list_A[idx].split('/')[-1].replace('.tif', '.png')
        img_B = io.imread(self.imgs_list_B[idx])
        transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.img_transform = transforms.Compose(transform_list)
        label = self.labels[idx] // 255
        if self.mode == "train":
            [img_A, img_B], [label] = self.augm.transform([img_A, img_B], [label])
        img_A = self.img_transform(img_A)
        img_B = self.img_transform(img_B)
        # img_A = F.to_tensor(img_A)
        # img_B = F.to_tensor(img_B)
        label = torch.from_numpy(np.array(label, np.uint8)).unsqueeze(dim=0)
        # print(label.shape)
        # img_B = img_B/255
        # img_A = img_A/255  

        return img_A, img_B, label.squeeze(), name

    def __len__(self):
        return len(self.imgs_list_A)
# class Data_test(data.Dataset):
#     def __init__(self, test_dir):
#         self.imgs_A = []
#         self.imgs_B = []
#         self.mask_name_list = []
#         imgA_dir = os.path.join(test_dir, 'pre')
#         imgB_dir = os.path.join(test_dir, 'post')
#         data_list = os.listdir(imgA_dir)
#         for it in data_list:
#             if (it[-4:]=='.png'):
#                 img_A_path = os.path.join(imgA_dir, it)
#                 img_B_path = os.path.join(imgB_dir, it)
#                 self.imgs_A.append(io.imread(img_A_path))
#                 self.imgs_B.append(io.imread(img_B_path))
#                 self.mask_name_list.append(it)
#         self.len = len(self.imgs_A)

#     def get_mask_name(self, idx):
#         return self.mask_name_list[idx]

#     def __getitem__(self, idx):
#         img_A = self.imgs_A[idx]
#         img_B = self.imgs_B[idx]
#         img_A = normalize_image(img_A, 'A')
#         img_B = normalize_image(img_B, 'B')
#         return F.to_tensor(img_A), F.to_tensor(img_B)

#     def __len__(self):
#         return self.len
