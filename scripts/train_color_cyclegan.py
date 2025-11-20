import torch
from torch.utils.data import DataLoader
from my_datasets.color_dataset import UnalignedColorDataset , UnalignedColorDatasetAug
from model.color_cyclegan_model import ColorCycleGANModel
import argparse
import os
from replay_buffer import ImageBuffer

fake_A_buffer = ImageBuffer(max_size=50)
fake_B_buffer = ImageBuffer(max_size=50)
# ---------------------------
# 1. Arguments
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='../data/', help='folder with trainA and trainB')
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lambda_cycle', type=float, default=10.0)
parser.add_argument('--lambda_identity', type=float, default=128.0)
parser.add_argument('--save_dir', type=str, default='../results', help='folder to save results and checkpoints')
parser.add_argument('--isTrain' ,action='store_true', help='True for training, False for testing')
parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='folder to save checkpoints')
parser.add_argument('--name', type=str, default='color_cyclegan_experiment', help='name of the experiment')
parser.add_argument('--preprocess', type=str, default='none', help='scaling/cropping of images')
parser.add_argument('--input_nc', type=int, default=2, help='number of input channels')
parser.add_argument('--output_nc', type=int, default=2, help='number of output channels')
parser.add_argument('--ngf', type=int, default=8, help='number of generator filters')
parser.add_argument('--ndf', type=int, default=16, help='number of discriminator filters')
parser.add_argument('--use_dropout', action='store_true', help='enable dropout in generator')
parser.add_argument('--augment-dataset', default=False, help='use data augmentation in the dataset') # usually loss functions gives worse results with augmentation

opt = parser.parse_args()
opt.isTrain = True
opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(opt.save_dir, exist_ok=True)
# ---------------------------
# 2. Dataset and DataLoader
# ---------------------------
if opt.augment_dataset:
    dataset = UnalignedColorDatasetAug(root_dir=opt.dataroot, image_size=opt.image_size)
else:
    dataset = UnalignedColorDataset(root_dir=opt.dataroot, image_size=opt.image_size)

dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

# ---------------------------
# 3. Model
# ---------------------------
model = ColorCycleGANModel(opt)
model.netG_A.to(opt.device)
model.netG_B.to(opt.device)
model.netD_A.to(opt.device)
model.netD_B.to(opt.device)

# Optimizers
optimizer_G = torch.optim.Adam(
    list(model.netG_A.parameters()) + list(model.netG_B.parameters()),
    lr=opt.lr, betas=(0.5, 0.999)
)
optimizer_D = torch.optim.Adam(
    list(model.netD_A.parameters()) + list(model.netD_B.parameters()),
    lr=opt.lr, betas=(0.5, 0.999)
)

# ---------------------------
# 4. Training
# ---------------------------

min_loss = float('inf')
for epoch in range(opt.epochs):
    for i, data in enumerate(dataloader):
        AB_A = data['AB_A'].to(opt.device)
        AB_B = data['AB_B'].to(opt.device)
        L_A = data['L_A'].to(opt.device)
        L_B = data['L_B'].to(opt.device)

        # --- Forward ---
        model.forward(AB_A, AB_B)

        # --- Backward Generators ---
        optimizer_G.zero_grad()
        model.backward_G(AB_A, AB_B)
        optimizer_G.step()

        # --- Backward Discriminators ---
        optimizer_D.zero_grad()
        model.backward_D(AB_A, AB_B, fake_A_buffer=fake_A_buffer, fake_B_buffer=fake_B_buffer)
        optimizer_D.step()

        # --- Logging ---

        print(f"[Epoch {epoch + 1}/{opt.epochs}] [Batch {i}/{len(dataloader)}] "
            f"Loss_G: {model.loss_G.item():.4f} "
            f"Loss_D_A: {model.loss_D_A.item():.4f} "
            f"Loss_D_B: {model.loss_D_B.item():.4f}")


    # --- Save checkpoint every epoch ---
    # torch.save({
    #     'opt': vars(opt),
    #     'netG_A': model.netG_A.state_dict(),
    #     'netG_B': model.netG_B.state_dict(),
    #     'netD_A': model.netD_A.state_dict(),
    #     'netD_B': model.netD_B.state_dict(),
    #     'optimizer_G': optimizer_G.state_dict(),
    #     'optimizer_D': optimizer_D.state_dict()
    # }, os.path.join(opt.save_dir, f'checkpoint_epoch_{epoch + 1}.pth'))

    if model.loss_G.item() < min_loss:
        min_loss = model.loss_G.item()
        torch.save({
            'opt': vars(opt),
            'netG_A': model.netG_A.state_dict(),
            'netG_B': model.netG_B.state_dict(),
            'netD_A': model.netD_A.state_dict(),
            'netD_B': model.netD_B.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'optimizer_D': optimizer_D.state_dict()
        }, os.path.join(opt.save_dir, 'best_checkpoint.pth'))
        print(f"Best model saved at epoch {epoch + 1} with Loss_G: {min_loss:.4f}")

print("Training finished!")
