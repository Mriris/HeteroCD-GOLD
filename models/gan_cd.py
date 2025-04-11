from .DualEUNet import DualEUNet
from .base_model import BaseModel
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
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.change_pred = None
        self.isTrain = is_train
        self.n_epochs_gen = opt.n_epochs_gen
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = DualEUNet(opt.input_nc, opt.output_nc)
        # im                  = torch.zeros(1, 3, 256,256).to('cpu')  # image size(1, 3, 512, 512) BCHW
        # input_layer_names   = ["images"]
        # output_layer_names  = ["output"]

        # # Export the model

        # torch.onnx.export(self.netG,
        #                 im,
        #                 f               = "bbb.onnx",
        #                 verbose         = False,
        #                 opset_version   = 12,
        #                 training        = torch.onnx.TrainingMode.EVAL,
        #                 do_constant_folding = True,
        #                 input_names     = input_layer_names,
        #                 output_names    = output_layer_names,
        #                 )
        # networks.init_weights(self.netG.gen_decoder, opt.init_type, init_gain= opt.init_gain)
        # networks.init_weights(self.netG.cd_decoder, opt.init_type, init_gain= opt.init_gain)
        # networks.init_weights(self.netG.discriminator, opt.init_type, init_gain= opt.init_gain)
        # networks.init_weights(self.netG.cls_seg, opt.init_type, init_gain= opt.init_gain)
        self.netG.to(opt.gpu_ids[0])
        self.is_train = is_train
        if is_train:
            self.netG = torch.nn.DataParallel(self.netG, opt.gpu_ids)  # multi-GPUs

        # self.netG = networks.define_G(opt.input_nc, opt.output_nc, 64, "unet_256", "batch",
        #                               not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc * 2, opt.ndf, opt.netD,
                                          opt.n_layers_D, "batch", opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionCosine = torch.nn.CosineSimilarity(dim=1)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.netG.parameters()), lr=opt.lr,
                                                 betas=(0.9, 0.999), weight_decay=0.01)
            self.optimizer_CD = torch.optim.AdamW(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.AdamW(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            # self.optimizers.append(self.optimizer_CD)

    def set_input(self, A, B, label, name, device):
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
        self.netG.load_state_dict(checkpoint)
        if not self.isTrain:
            self.netG.eval()

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        [self.fake_B, self.fake_BB] = self.netG(self.real_A, self.real_B)  # G(A)

    def forward_CD(self):
        self.change_pred = self.netG(self.opt_img, self.sar_img)  # G(A)
        # if self.is_train:
        #     # print(self.label.size(),self.opt_gen.size())
        #     mask = F.interpolate(self.label.unsqueeze(1), size=(self.opt_gen.size(2),self.opt_gen.size(3)), mode='nearest')
        #     self.fake_B = self.opt_gen*(1-mask)
        #     self.fake_BB = self.sar_gen*(1-mask)
        #     self.real_B = F.interpolate(self.sar_img, size=(self.opt_gen.size(2),self.opt_gen.size(3)), mode='nearest')*(1-mask)
        #     self.real_A = F.interpolate(self.opt_img, size=(self.opt_gen.size(2),self.opt_gen.size(3)), mode='nearest')*(1-mask)

    def get_val_pred(self):
        self.netG.eval()
        self.is_train = False
        with torch.no_grad():
            self.forward_CD()
            cls_weights = torch.tensor([0.2, 0.8]).cuda()
            loss_bn = CE_Loss(self.change_pred, self.label, cls_weights)
        self.is_train = True
        return self.change_pred, loss_bn

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""

        fake_AB = torch.cat((self.real_A, self.fake_B),
                            1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        fake_ABB = torch.cat((self.real_A, self.fake_BB),
                             1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fakeB = self.netD(fake_ABB.detach())
        self.loss_D_fake = (self.criterionGAN(pred_fake, False) + self.criterionGAN(pred_fakeB, False)) / 2
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

        # mask = F.interpolate(self.label.unsqueeze(1), size=(self.opt_gen.size(2),self.opt_gen.size(3)), mode='nearest')
        # self.fake_B = self.opt_gen*(1-mask)
        # self.real_B = self.opt_sar*(1-mask)
        # pred_fake = self.netD(self.fake_B)
        # self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # # Real
        # pred_real = self.netD(self.real_B)
        # self.loss_D_real = self.criterionGAN(pred_real, True)
        # # combine loss and calculate gradients
        # self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        # self.loss_D.backward()

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
        self.change_pred = F.interpolate(self.change_pred, size=(self.opt_img.size(2), self.opt_img.size(3)),
                                         mode='bilinear', align_corners=True)
        cls_weights = torch.tensor([0.2, 0.8]).cuda()
        self.label = self.label.long()
        self.loss_CD = CE_Loss(self.change_pred, self.label, cls_weights=cls_weights) * 100 + Dice_loss(
            self.change_pred, self.label) * 100

        # combine loss and calculate gradients
        self.loss_G = self.loss_CD
        self.loss_G.backward()

    def backward_CD(self):
        """Calculate GAN and L1 loss for the generator"""
        cls_weights = torch.tensor([1.0, 1.0]).cuda()
        self.loss_CD = CE_Loss(self.change_pred, self.label, cls_weights=cls_weights)
        self.loss_CD.backward()

    def optimize_parameters(self, epoch):
        self.netG.train()
        self.forward_CD()
        # self.epoch = epoch                 # compute fake images: G(A)
        # # update D
        # self.set_requires_grad(self.netD, True)  # enable backprop for D
        # self.optimizer_D.zero_grad()     # set D's gradients to zero
        # self.backward_D()                # calculate gradients for D
        # self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()
        return self.change_pred
        # def optimize_parameters(self,epoch):
    #     self.epoch = epoch
    #     self.forward_CD()
    #     # self.set_requires_grad(self.netG.module.cd_decoder, False)
    #     # self.set_requires_grad(self.netD, True)  # enable backprop for D
    #     # self.optimizer_D.zero_grad()     # set D's gradients to zero
    #     # self.backward_D()                # calculate gradients for D
    #     # self.optimizer_D.step()          # update D's weights
    #     # update G

    #     self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
    #     self.optimizer_G.zero_grad()        # set G's gradients to zero
    #     self.backward_G()                 # calculate graidents for G
    #     self.optimizer_G.step()             # update G's weights
    # freeze the decoder
    # self.optimizer_CD.zero_grad()
    # self.backward_CD()
    # self.optimizer_CD.step()
    # self.optimizer_CD.step()
    # def forward(self):
    #     """Run forward pass; called by both functions <optimize_parameters> and <test>."""
    #     # [self.fake_B1,self.fake_B2] = self.netG(self.real_A,self.real_B)  # G(A)
    #     self.fake_B1 = self.netG(self.real_A)  # G(A)

    # def backward_D(self):
    #     """Calculate GAN loss for the discriminator"""
    #     # Fake; stop backprop to the generator by detaching fake_B
    #     fake_AB1 = torch.cat((self.real_A, self.fake_B1), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
    #     # fake_AB2 = torch.cat((self.real_A, self.fake_B2), 1)
    #     pred_fake1 = self.netD(fake_AB1.detach())
    #     # pred_fake2 = self.netD(fake_AB2.detach())
    #     self.loss_D_fake = (self.criterionGAN(pred_fake1, False))
    #     # Real
    #     real_AB = torch.cat((self.real_A, self.real_B), 1)
    #     pred_real = self.netD(real_AB)
    #     self.loss_D_real = self.criterionGAN(pred_real, True)
    #     # combine loss and calculate gradients
    #     self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
    #     self.loss_D.backward()

    # def backward_G(self):
    #     """Calculate GAN and L1 loss for the generator"""
    #     # First, G(A) should fake the discriminator
    #     fake_AB1 = torch.cat((self.real_A, self.fake_B1), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
    #     # fake_AB2 = torch.cat((self.real_A, self.fake_B2), 1)
    #     pred_fake1 = self.netD(fake_AB1.detach())
    #     # pred_fake2 = self.netD(fake_AB2.detach())
    #     self.loss_G_GAN = (self.criterionGAN(pred_fake1, True))
    #     # Second, G(A) = B
    #     self.loss_G_L1 = (self.criterionL1(self.fake_B1, self.real_B) * 100)
    #     # combine loss and calculate gradients
    #     self.loss_G = self.loss_G_GAN + self.loss_G_L1
    #     self.loss_G.backward()

    # def optimize_parameters(self):
    #     self.forward()                   # compute fake images: G(A)
    #     # update D
    #     self.set_requires_grad(self.netD, True)  # enable backprop for D
    #     self.optimizer_D.zero_grad()     # set D's gradients to zero
    #     self.backward_D()                # calculate gradients for D
    #     self.optimizer_D.step()          # update D's weights
    #     # update G
    #     self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
    #     self.optimizer_G.zero_grad()        # set G's gradients to zero
    #     self.backward_G()                   # calculate graidents for G
    #     self.optimizer_G.step()             # udpate G's weights
