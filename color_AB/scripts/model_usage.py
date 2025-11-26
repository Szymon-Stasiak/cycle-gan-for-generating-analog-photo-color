import torch
from model.color_cyclegan_model_AB import ColorCycleGANModel
import matplotlib
matplotlib.use('TkAgg')
from utils.color_utils import rgb_to_lab_tensor, lab_tensor_to_rgb
from PIL import Image
import torchvision.transforms.functional as TF
import argparse

# ==================================
# 1. FORCE CPU
# ==================================
device = torch.device("cpu")
print("Running on:", device)

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


# ==================================
# 2. Load checkpoint (force CPU)
# ==================================

checkpoint_path = '../results/best_checkpoint.pth'
checkpoint = torch.load(checkpoint_path, map_location="cpu")

saved_opt_dict = checkpoint['opt']
opt = argparse.Namespace(**saved_opt_dict)

model = ColorCycleGANModel(opt)

# load generators / discriminators
model.netG_A.load_state_dict(checkpoint['netG_A'])
model.netG_B.load_state_dict(checkpoint['netG_B'])
model.netD_A.load_state_dict(checkpoint['netD_A'])
model.netD_B.load_state_dict(checkpoint['netD_B'])

# force CPU for model
model.to(device)
model.netG_A.eval()
model.netG_B.eval()

print(f"Checkpoint {checkpoint_path} loaded successfully on CPU!")


# ==================================
# 3. Load image
# ==================================
image_path = '../../data/1000000544.jpg'
img = Image.open(image_path).convert("RGB")
img = pad_to_multiple(img, 8)

L_tensor, AB_tensor = rgb_to_lab_tensor(img)
AB_tensor_batch = AB_tensor.unsqueeze(0).to(device)

print("Input AB batch shape:", AB_tensor_batch.shape)


# ==================================
# 4. Run model
# ==================================
with torch.no_grad():
    fake_AB_batch = model.transform_to_analog(AB_tensor_batch)

fake_AB = fake_AB_batch[0].cpu()
fake_AB = torch.clamp(fake_AB, -1.0, 1.0)
print("Fake AB shape:", fake_AB.shape)


# ==================================
# 5. Crop and convert to RGB
# ==================================
H, W = L_tensor.shape[1:]
fake_AB = fake_AB[:, :H, :W]

rgb_fake = lab_tensor_to_rgb(L_tensor, fake_AB)
rgb_fake.show()

print("DONE.")
