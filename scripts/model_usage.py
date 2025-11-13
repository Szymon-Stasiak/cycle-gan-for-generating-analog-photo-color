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

# -----------------------------
# 1. Setup model options
# -----------------------------
opt = Namespace(
    input_nc=2,   # AB channels
    output_nc=2,  # AB channels
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
checkpoint_path = '../results/checkpoint_epoch_3.pth'

checkpoint = torch.load(checkpoint_path, map_location=opt.device)
model.netG_A.load_state_dict(checkpoint['netG_A'])
model.netG_B.load_state_dict(checkpoint['netG_B'])
model.eval()  # evaluation mode (BatchNorm/Dropout)

# -----------------------------
# 3. Load and preprocess image
# -----------------------------
image_path = '../data/1000000544.jpg'
img = Image.open(image_path).convert("RGB")

L_tensor, AB_tensor = rgb_to_lab_tensor(img)  # L: [1,H,W], AB: [2,H,W]

# Add batch dimension for generator
AB_tensor_batch = AB_tensor.unsqueeze(0)  # [1,2,H,W]
print("Input AB tensor batch shape:", AB_tensor_batch.shape)

# Move tensor to same device as generator
device = next(model.netG_A.parameters()).device
AB_tensor_batch = AB_tensor_batch.to(device)

# -----------------------------
# 4. Transform AB -> analog
# -----------------------------
with torch.no_grad():
    fake_AB_batch = model.transform_to_analog(AB_tensor_batch)

# Remove batch dimension
fake_AB = fake_AB_batch[0].cpu()  # [2,H,W]
fake_AB = torch.clamp(fake_AB, -1.0, 1.0)

print("Fake AB tensor shape:", fake_AB.shape)


rgb_fake = lab_tensor_to_rgb(L_tensor, fake_AB)

# Display result
rgb_fake.show()
