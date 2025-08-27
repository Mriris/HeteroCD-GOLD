import glob
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.TripleEUNet import DualEUNet
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def init():
    net = DualEUNet(3,3).cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('/data/jingwei/HeteCD/HeteGAN/checkpoints/FreqHete_cosine_new1/best_net_CD.pth', map_location=device)
    saved_weights = checkpoint
    new_state_dict = {}
    for k, v in saved_weights.items():
        if k.startswith('module.'):
            name = k[7:]  # remove the "module." prefix
        else:
            name = k
        # print(name)
        new_state_dict[name] = v

    # create a new model and load the new state dict
    net.load_state_dict(new_state_dict)
    net = net.eval()
    return net

class SemanticSegmentationTarget:
    """wrap the model.

    requirement: pip install grad-cam

    Args:
        category (int): Visualization class.
        mask (ndarray): Mask of class.
        size (tuple): Image size.
    """

    def __init__(self, category, mask, size):
        self.category = category
        self.mask = torch.from_numpy(mask)
        self.size = size
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        model_output = torch.unsqueeze(model_output, dim=0)
        # print(model_output.shape)
        model_output = F.interpolate(
            model_output, size=self.size, mode='bilinear')
        
        model_output = torch.squeeze(model_output, dim=0)
        print(model_output.shape)
        print(self.mask.shape)
        return (model_output[self.category, :, :] * self.mask).sum()


def main():

    # checkpoint = torch.load('logs/changeECD_r18_levir_test2/best_mIoU_iter_39000.pth', map_location='cuda:0')
    # saved_weights = checkpoint['state_dict']
    # new_state_dict = {}
    # for k, v in saved_weights.items():
    #     if k.startswith('module.'):
    #         name = k[7:]  # remove the "module." prefix
    #     else:
    #         name = k
    #     print(name)

    model = init()

    img_pathsA = glob.glob(os.path.join("/data/jingwei/HeteCD/data/xiongan_data/test/A", "*.png"))
    img_pathsB = glob.glob(os.path.join("/data/jingwei/HeteCD/data/xiongan_data/test/B", "*.png"))
    for i in range(len(img_pathsA)):
        img_pathA = img_pathsA[i]
        img_pathB = img_pathsB[i]
        print(i)
        if 'win' in sys.platform:
            imgname = img_pathA.split('.')[0].split('\\')[-1]
        else:
            imgname = img_pathA.split('.')[0].split('/')[-1]
        imageA = np.asarray(Image.open(img_pathA).convert('RGB'))
        imageB = np.asarray(Image.open(img_pathB).convert('RGB'))
        transform_list = [transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        img_transform = transforms.Compose(transform_list)
        img1 = img_transform(imageA)
        img2 = img_transform(imageB)
        img1 = img1.unsqueeze(0).cuda()
        img2 = img2.unsqueeze(0).cuda()
        # img1 = TF.to_tensor(imageA).cuda().unsqueeze(0)
        # img2 = TF.to_tensor(imageB).cuda().unsqueeze(0)
        model = model.cuda()
        output = model(img1, img2)
        output = F.interpolate(output, size=(512, 512), mode='bilinear')
        output = output[0].argmax(dim=0).cpu().numpy().astype(np.uint8)

    # result data conversion
    # prediction_data = result.pred_sem_seg.data
    # pre_np_data = prediction_data.cpu().numpy().squeeze(0)

        target_layers = model.discriminator.fc2,
        # target_layers = [eval(f'model.{target_layers}')]
        print(target_layers)

        category = 1
        mask_float = np.float32(output == category)
        print(mask_float.shape)
        # # data processing

        # Grad CAM(Class Activation Maps)
        # Can also be LayerCAM, XGradCAM, GradCAMPlusPlus, EigenCAM, EigenGradCAM
        rgb_img = np.float32(imageA) / 255
        targets = [
            SemanticSegmentationTarget(category, mask_float, (512, 512))
        ]
        with GradCAM(
                model=model,
                target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=[img1,img2], targets=targets)[0, :]
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            # save cam file
            savenmae = os.path.join('freqffn2_before', imgname + '_cam.png')
            if os.path.exists('freqffn2_before') is False:
                os.makedirs('freqffn2_before')
            Image.fromarray(cam_image).save(savenmae)


if __name__ == '__main__':
    main()
    
