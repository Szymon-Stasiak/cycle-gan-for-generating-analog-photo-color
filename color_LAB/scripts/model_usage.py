import torch
from model.color_cyclegan_model_LAB import ColorCycleGANModel
import matplotlib
matplotlib.use('TkAgg')
from utils.color_utils import rgb_to_lab_tensor, lab_tensor_to_rgb
from PIL import Image
import torchvision.transforms.functional as TF
import argparse


device = torch.device("cpu")
print("Running only on CPU.")


# =====================================================================================
# Padding helper
# =====================================================================================
def pad_to_multiple(img, multiple=8):
    w, h = img.size
    new_w = (w + multiple - 1) // multiple * multiple
    new_h = (h + multiple - 1) // multiple * multiple
    pad_w = new_w - w
    pad_h = new_h - h
    return TF.pad(img, (0, 0, pad_w, pad_h))


# =====================================================================================
# 2. Load model + checkpoint (force CPU mapping)
# =====================================================================================
checkpoint_path = '../results/best_checkpoint.pth'
checkpoint = torch.load(checkpoint_path, map_location="cpu")

saved_opt_dict = checkpoint['opt']
opt = argparse.Namespace(**saved_opt_dict)

model = ColorCycleGANModel(opt)

model.netG_A.load_state_dict(checkpoint['netG_A'])
model.netG_B.load_state_dict(checkpoint['netG_B'])
model.netD_A.load_state_dict(checkpoint['netD_A'])
model.netD_B.load_state_dict(checkpoint['netD_B'])

model.to(device)
model.netG_A.eval()
model.netG_B.eval()

print(f"Checkpoint {checkpoint_path} loaded successfully on CPU!")


# =====================================================================================
# 3. Load and preprocess input image
# =====================================================================================
image_path = '../../data/1000000544.jpg'
img = Image.open(image_path).convert("RGB")
img = pad_to_multiple(img, 8)

L_tensor, AB_tensor = rgb_to_lab_tensor(img)        # L: [1,H,W], AB: [2,H,W]
lab_tensor = torch.cat([L_tensor, AB_tensor], dim=0)  # LAB: [3,H,W]

lab_batch = lab_tensor.unsqueeze(0)  # → [1,3,H,W]
lab_batch = lab_batch.to(device)


# =====================================================================================
# 4. Run inference (generator G_A)
# =====================================================================================
with torch.no_grad():
    fake_lab_batch = model.inference(lab_batch)

fake_lab = fake_lab_batch[0].cpu()
fake_lab = torch.clamp(fake_lab, -1.0, 1.0)

L = fake_lab[0:1]   # [1,H,W]
AB = fake_lab[1:3]  # [2,H,W]

rgb_fake = lab_tensor_to_rgb(L, AB)
rgb_fake.show()

print("DONE.")
