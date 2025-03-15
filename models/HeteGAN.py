import torch
from .base_model import BaseModel
from . import networks
from .DualEUNet import DualEUNet
from .loss import *
class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    # @staticmethod
    # def modify_commandline_options(parser, is_train=True):
    #     """Add new dataset-specific options, and rewrite default values for existing options.

    #     Parameters:
    #         parser          -- original option parser
    #         is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

    #     Returns:
    #         the modified parser.

    #     For pix2pix, we do not use image buffer
    #     The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
    #     By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
    #     """


    #     return parser

    def __init__(self, opt, is_train=True):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt, is_train=True)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake','CD']
        self.loss_names = ['CD']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.change_pred = None
        self.isTrain = is_train
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['CD']
        # define networks (both generator and discriminator)
        self.netCD = DualEUNet(3,2)
        # for i in range(1000):
        #     # 创建输入网络的tensor
        #     # from fvcore.nn import FlopCountAnalysis, parameter_count_table

        #     # tensor = (torch.rand(1, 3, 512, 512).float().cuda(),torch.rand(1, 3, 512, 512).float().cuda())

        #     # # 分析FLOPs
        #     # flops = FlopCountAnalysis(self.net_G, tensor)
        #     # print("FLOPs: ", flops.total())
        #     # from ptflops import get_model_complexity_info
        #     # macs, params = get_model_complexity_info(self.net_G , (3, 512, 512), as_strings=True,
        #     #                                     print_per_layer_stat=True, verbose=True)
        #     # print(macs)
        #     # print(params)


        #     from thop import profile
        #     input = torch.randn(1, 3, 512, 512).float().cuda()
        #     flops, params = profile(self.netCD.cuda(), inputs=(input, input,))
            
        #     from thop import clever_format
        #     flops, params = clever_format([flops, params], "%.3f")
        #     print('flops:{}'.format(flops))
        #     print('params:{}'.format(params))
        #     print(f"Current memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        #     print(f"Max memory usage: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        self.netCD.to(opt.gpu_ids[0])
        self.is_train = is_train
        if is_train:
            self.netCD = torch.nn.DataParallel(self.netCD, opt.gpu_ids)  # multi-GPUs
        
        # self.netG = networks.define_G(opt.input_nc, opt.output_nc, 64, "unet_256", "batch",
        #                               not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:

            self.optimizer_G = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.netCD.parameters()), lr=opt.lr, betas=(0.9, 0.999),weight_decay=0.01 )
            self.optimizers.append(self.optimizer_G)

    def set_input(self, A, B, label, name,device):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
    
        self.opt_img = A.to(device)
        self.sar_img = B.to(device)
        self.label = label.to(device)
        # self.real_A = self.opt_img*(1-self.label.unsqueeze(1))
        # self.real_B = self.sar_img*(1-self.label.unsqueeze(1))
        self.name = name

    def load_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        for key in list(checkpoint.keys()):
            if key.startswith('module.'):
                checkpoint[key[7:]] = checkpoint[key]
                del checkpoint[key]
        self.netCD.load_state_dict(checkpoint)
        if not self.isTrain:
            self.netCD.eval()
            
        

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        [self.fake_B, self.fake_BB] = self.netCD(self.real_A,self.real_B)  # G(A)
    def get_val_pred(self):
        self.netCD.eval()
        self.is_train = False
        with torch.no_grad():
            self.forward_CD()
            cls_weights = torch.tensor([0.2, 0.8]).cuda()
            loss_bn = CE_Loss(self.change_pred, self.label,cls_weights)
        self.is_train = True
        return self.change_pred, loss_bn
    
    def forward_CD(self):
        self.change_pred = self.netCD(self.opt_img,self.sar_img)  # G(A)
    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        # pred_fake = self.netD(fake_AB)
        # fake_ABB = torch.cat((self.real_A, self.fake_BB), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        # pred_fakeB = self.netD(fake_ABB)
        # self.loss_G_GAN = (self.criterionGAN(pred_fake, True) + self.criterionGAN(pred_fakeB, True))/2
        # # Second, G(A) = B
        # self.loss_G_L1 = (self.criterionL1(self.fake_B, self.real_B) + self.criterionL1(self.fake_BB, self.real_B))/10
        # # self.loss_G_L1 = (1-self.criterionCosine (self.fake_B, self.real_B) + 1-self.criterionCosine (self.fake_BB, self.real_B))/2
        # # combine loss and calculate gradients
        # self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.change_pred = F.interpolate(self.change_pred, size=(self.opt_img.size(2),self.opt_img.size(3)), mode='bilinear', align_corners=True)
        cls_weights = torch.tensor([0.2, 0.8]).cuda()
        self.label = self.label.long()
        self.loss_CD = CE_Loss(self.change_pred, self.label, cls_weights=cls_weights)*100 + Dice_loss(self.change_pred, self.label)*100

        # combine loss and calculate gradients
        self.loss_G = self.loss_CD
        self.loss_G.backward()


    def optimize_parameters(self,epoch):
        self.netCD.train()
        self.forward_CD()  
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()
        return self.change_pred       
