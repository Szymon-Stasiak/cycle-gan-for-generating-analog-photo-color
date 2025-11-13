import torch
from torch import nn
from model.base_model import BaseModel
from model.networks import ResnetGenerator, NLayerDiscriminator, GANLoss


class ColorCycleGANModel(BaseModel):
    """
    CycleGAN modified for learning color transformations (AB channels)
    """

    def __init__(self, opt):
        super().__init__(opt)

        # Generators: digital -> analog, analog -> digital
        self.netG_A = ResnetGenerator(input_nc=2, output_nc=2, ngf=64)
        self.netG_B = ResnetGenerator(input_nc=2, output_nc=2, ngf=64)

        # Discriminators
        self.netD_A = NLayerDiscriminator(input_nc=2, ndf=64, n_layers=3)
        self.netD_B = NLayerDiscriminator(input_nc=2, ndf=64, n_layers=3)

        # GAN criterion
        self.criterionGAN = GANLoss(gan_mode='vanilla')
        self.criterionCycle = nn.L1Loss()
        self.criterionIdt = nn.L1Loss()

    def forward(self, AB_A, AB_B):

        """
        AB_A: batch from digital
        AB_B: batch from analog
        """
        # Generators
        self.fake_B = self.netG_A(AB_A)  # A -> B
        self.rec_A = self.netG_B(self.fake_B)  # B -> A (cycle)

        self.fake_A = self.netG_B(AB_B)  # B -> A
        self.rec_B = self.netG_A(self.fake_A)  # A -> B (cycle)

    def backward_G(self, AB_A, AB_B):
        # Identity loss
        idt_B = self.netG_A(AB_B)
        idt_A = self.netG_B(AB_A)
        self.loss_idt = self.criterionIdt(idt_B, AB_B) * self.opt.lambda_identity + \
                        self.criterionIdt(idt_A, AB_A) * self.opt.lambda_identity

        # GAN loss
        self.loss_G_A = self.criterionGAN(self.netD_B(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_A(self.fake_A), True)
        self.loss_GAN = self.loss_G_A + self.loss_G_B

        # Cycle loss
        self.loss_cycle = self.criterionCycle(self.rec_A, AB_A) * self.opt.lambda_cycle + \
                          self.criterionCycle(self.rec_B, AB_B) * self.opt.lambda_cycle

        # Total generator loss
        self.loss_G = self.loss_GAN + self.loss_cycle + self.loss_idt
        self.loss_G.backward()

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Total
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D(self, AB_A, AB_B):
        self.loss_D_A = self.backward_D_basic(self.netD_A, AB_A, self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, AB_B, self.fake_B)

    def set_input(self, input):
        """Move input batch to device and store as model attributes"""
        device = next(self.parameters()).device  # get model device
        self.AB_A = input['AB_A'].to(device)
        self.AB_B = input['AB_B'].to(device)
        self.L_A = input['L_A'].to(device)
        self.L_B = input['L_B'].to(device)

    def optimize_parameters(self, optimizer_G, optimizer_D):
        """Perform forward pass and update generators and discriminators"""
        # Forward
        self.forward(self.AB_A, self.AB_B)

        # Generators
        optimizer_G.zero_grad()
        self.backward_G(self.AB_A, self.AB_B)
        optimizer_G.step()

        # Discriminators
        optimizer_D.zero_grad()
        self.backward_D(self.AB_A, self.AB_B)
        optimizer_D.step()

    def transform_to_analog(self, AB_A):
        """
        Processes a digital image (AB_A) to analog style (B) using generator netG_A.

        AB_A: image tensor in format [1, 1, H, W] or batch [B, 1, H, W]
        Returns: tensor of the processed image in the same format
        """
        self.eval()  # evaluation mode
        device = next(self.netG_A.parameters()).device  # use generator, not model
        if isinstance(AB_A, torch.Tensor):
            input_tensor = AB_A.to(device)
        else:
            raise TypeError("AB_A must be a PyTorch tensor")

        with torch.no_grad():
            fake_B = self.netG_A(input_tensor)
        return fake_B
