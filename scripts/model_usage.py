import torch
import cv2
import numpy as np
from model.color_cyclegan_model import ColorCycleGANModel
from argparse import Namespace
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from utils.color_utils import rgb_to_lab_tensor, lab_tensor_to_rgb
from PIL import Image
import torchvision.transforms.functional as TF

# -----------------------------
# Padding helper
# -----------------------------
def pad_to_multiple(img, multiple=8):
    w, h = img.size
    new_w = (w + multiple - 1) // multiple * multiple
    new_h = (h + multiple - 1) // multiple * multiple
    pad_w = new_w - w
    pad_h = new_h - h
    return TF.pad(img, (0, 0, pad_w, pad_h))


# -----------------------------
# 1. Setup model options
# -----------------------------
opt = Namespace(
    input_nc=2,
    output_nc=2,
    ngf=64,
    ndf=64,
    no_dropout=False,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    save_dir='../results',
    isTrain=False,
    checkpoints_dir='checkpoints',
    name='color_cyclegan_experiment',
    preprocess='resize_and_crop',
    lambda_identity=0.5,
    lambda_cycle=10.0
)

# -----------------------------
# 2. Load model and checkpoint
# -----------------------------
model = ColorCycleGANModel(opt)
checkpoint_path = '../results/checkpoint_epoch_10.pth'
checkpoint = torch.load(checkpoint_path, map_location=opt.device)
model.netG_A.load_state_dict(checkpoint['netG_A'])
model.netG_B.load_state_dict(checkpoint['netG_B'])
model.eval()

# -----------------------------
# 3. Load and preprocess image
# -----------------------------
image_path = './my_datasets/p2.jpg'
img = Image.open(image_path).convert("RGB")
img = pad_to_multiple(img, 8)

L_tensor, AB_tensor = rgb_to_lab_tensor(img)
AB_tensor_batch = AB_tensor.unsqueeze(0)
print("Input AB tensor batch shape:", AB_tensor_batch.shape)

device = next(model.netG_A.parameters()).device
AB_tensor_batch = AB_tensor_batch.to(device)

# -----------------------------
# 4. Transform AB
# -----------------------------
with torch.no_grad():
    fake_AB_batch = model.transform_to_analog(AB_tensor_batch)

fake_AB = fake_AB_batch[0].cpu()
fake_AB = torch.clamp(fake_AB, -1.0, 1.0)
print("Fake AB tensor shape:", fake_AB.shape)

# -----------------------------
# 5. Crop fake_AB to match L_tensor
# -----------------------------
H, W = L_tensor.shape[1:]
fake_AB = fake_AB[:, :H, :W]

rgb_fake = lab_tensor_to_rgb(L_tensor, fake_AB)

rgb_fake.show()
rgb_fake.save('./my_datasets/p2_analog_output.jpg')
print("Saved analog output image to ./my_datasets/p2_analog_output.jpg")
