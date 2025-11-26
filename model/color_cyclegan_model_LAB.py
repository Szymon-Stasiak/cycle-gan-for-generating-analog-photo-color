import torch
from torch import nn
from model.base_model import BaseModel
from model.networks import ResnetGenerator, NLayerDiscriminator, GANLoss


class ColorCycleGANModel(BaseModel, nn.Module):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        nn.Module.__init__(self)
        super(ColorCycleGANModel, self).__init__(opt)

        self.opt = opt

        self.device = opt.device

        self.netG_A = ResnetGenerator(
            input_nc=opt.input_nc,
            output_nc=opt.output_nc,
            ngf=opt.ngf,
            use_dropout=opt.use_dropout
        )
        self.netG_B = ResnetGenerator(
            input_nc=opt.input_nc,
            output_nc=opt.output_nc,
            ngf=opt.ngf,
            use_dropout=opt.use_dropout
        )

        self.netD_A = NLayerDiscriminator(
            input_nc=opt.input_nc,
            ndf=opt.ndf,
            n_layers=3
        )
        self.netD_B = NLayerDiscriminator(
            input_nc=opt.input_nc,
            ndf=opt.ndf,
            n_layers=3
        )

        self.criterionGAN = GANLoss(gan_mode='vanilla')
        self.criterionCycle = nn.L1Loss()
        self.criterionIdt = nn.L1Loss()

    def forward(self):
        self.fake_B = self.netG_A(self.LAB_A)
        self.rec_A = self.netG_B(self.fake_B)

        self.fake_A = self.netG_B(self.LAB_B)
        self.rec_B = self.netG_A(self.fake_A)

    def backward_G(self):
        # Identity loss
        idt_B = self.netG_A(self.LAB_B)
        idt_A = self.netG_B(self.LAB_A)
        self.loss_idt = self.criterionIdt(idt_B, self.LAB_B) * self.opt.lambda_identity + \
                        self.criterionIdt(idt_A, self.LAB_A) * self.opt.lambda_identity

        # GAN loss
        self.loss_G_A = self.criterionGAN(self.netD_B(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_A(self.fake_A), True)
        self.loss_GAN = self.loss_G_A + self.loss_G_B

        # Cycle loss
        self.loss_cycle = self.criterionCycle(self.rec_A, self.LAB_A) * self.opt.lambda_cycle + \
                          self.criterionCycle(self.rec_B, self.LAB_B) * self.opt.lambda_cycle

        # Total generator loss
        self.loss_tv = tv_loss(self.fake_B) * 0.0
        self.loss_G = self.loss_GAN + self.loss_cycle + self.loss_idt + self.loss_tv
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

    def backward_D(self, fake_A_buffer=None, fake_B_buffer=None):
        if fake_A_buffer is not None:
            fake_A = fake_A_buffer.push_and_pop(self.fake_A)
        else:
            fake_A = self.fake_A

        if fake_B_buffer is not None:
            fake_B = fake_B_buffer.push_and_pop(self.fake_B)
        else:
            fake_B = self.fake_B

        self.loss_D_A = self.backward_D_basic(self.netD_A, self.LAB_A, fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.LAB_B, fake_B)

    def set_input(self, input):
        d = self.device
        self.LAB_A = torch.cat([input['L_A'], input['AB_A']], dim=1).to(d)
        self.LAB_B = torch.cat([input['L_B'], input['AB_B']], dim=1).to(d)
        self.L_A = input['L_A'].to(d)
        self.L_B = input['L_B'].to(d)


    def optimize_parameters(self, optimizer_G, optimizer_D, fake_A_buffer=None, fake_B_buffer=None):
        # forward
        self.forward()

        # G
        optimizer_G.zero_grad()
        self.backward_G()
        optimizer_G.step()

        # D
        optimizer_D.zero_grad()
        self.backward_D(fake_A_buffer, fake_B_buffer)
        optimizer_D.step()

    def transform_to_analog(self, LAB_A):
        self.eval()

        device = next(self.netG_A.parameters()).device

        if not isinstance(LAB_A, torch.Tensor):
            raise TypeError("LAB_A must be a PyTorch tensor")

        input_tensor = LAB_A.to(device)

        with torch.no_grad():
            fake_B = self.netG_A(input_tensor)

        return fake_B

    def inference(self, LAB_batch):
        self.eval()

        device = next(self.netG_A.parameters()).device
        LAB_batch = LAB_batch.to(device)

        with torch.no_grad():
            fake_LAB = self.netG_A(LAB_batch)

        return fake_LAB


def tv_loss(x):
    # x: [B, C, H, W]  (C=3 dla LAB)
    dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return dh + dw
