import torch
from torch import nn
from model.base_model import BaseModel
from model.networks import ResnetGenerator, NLayerDiscriminator, GANLoss


class ColorCycleGANModel(BaseModel):

    def __init__(self, opt):

        super(ColorCycleGANModel, self).__init__(opt)

        self.opt = opt

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
        # ensure loss buffers are on the correct device (moved after opt.device is set)
        try:
            self.criterionGAN.to(opt.device)
        except Exception:
            # opt.device might not be a torch.device yet; defer to caller
            pass
        self.criterionCycle = nn.L1Loss()
        self.criterionIdt = nn.L1Loss()

    def forward(self, AB_A, AB_B):

        self.fake_B = self.netG_A(AB_A)
        self.rec_A = self.netG_B(self.fake_B)

        self.fake_A = self.netG_B(AB_B)
        self.rec_B = self.netG_A(self.fake_A)

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

    def backward_D(self, AB_A, AB_B, fake_A_buffer=None, fake_B_buffer=None):
        # jeśli podano bufor, użyj go
        if fake_A_buffer is not None:
            fake_A = fake_A_buffer.push_and_pop(self.fake_A)
        else:
            fake_A = self.fake_A

        if fake_B_buffer is not None:
            fake_B = fake_B_buffer.push_and_pop(self.fake_B)
        else:
            fake_B = self.fake_B

        self.loss_D_A = self.backward_D_basic(self.netD_A, AB_A, fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, AB_B, fake_B)

    def set_input(self, input):
        # BaseModel doesn't inherit torch.nn.Module, so self.parameters() isn't available.
        # Use the device stored on the BaseModel (set from options) instead.
        device = getattr(self, "device", None) or self.opt.device

        # The dataset returns separated HSV channels (V, S, H) per domain.
        # Assemble them into 3-channel tensors expected by the generators.
        try:
            # Expect shapes like [B, 1, H, W] for each channel; concat on channel dim -> [B,3,H,W]
            self.AB_A = torch.cat([input['V_A'], input['S_A'], input['H_A']], dim=1).to(device)
            self.AB_B = torch.cat([input['V_B'], input['S_B'], input['H_B']], dim=1).to(device)
        except KeyError:
            # Fallback for datasets that already provide AB_A / AB_B tensors
            if 'AB_A' in input and 'AB_B' in input:
                self.AB_A = input['AB_A'].to(device)
                self.AB_B = input['AB_B'].to(device)
            else:
                raise KeyError("Dataset must provide either V_*/S_*/H_* channels or AB_A/AB_B tensors.")

        # Optional metadata
        self.A_path = input.get('A_path', None)
        self.B_path = input.get('B_path', None)
        self.image_paths = [self.A_path, self.B_path]
        # Luminance placeholders (not used by current dataset)
        self.L_A = None
        self.L_B = None

    def optimize_parameters(self, optimizer_G, optimizer_D):
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

        self.eval()  # evaluation mode
        device = next(self.netG_A.parameters()).device  # use generator, not model
        if isinstance(AB_A, torch.Tensor):
            input_tensor = AB_A.to(device)
        else:
            raise TypeError("AB_A must be a PyTorch tensor")

        with torch.no_grad():
            fake_B = self.netG_A(input_tensor)
        return fake_B


def tv_loss(x):
    # x: [B, C, H, W]  (C=2 for AB)
    dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return dh + dw
