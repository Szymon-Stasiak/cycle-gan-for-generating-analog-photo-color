import torch
from model.color_cyclegan_model import ColorCycleGANModel
from argparse import Namespace
import matplotlib
matplotlib.use('TkAgg')
from utils.color_utils import rgb_to_hsv_tensor, hsv_tensor_to_rgb
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
    dataroot='../data/',
    batch_size=10,
    image_size=256,
    lr=1e-5,
    epochs=30,
    lambda_cycle=32.0,
    lambda_identity=.0,
    save_dir='../results',
    isTrain=False,
    checkpoints_dir='checkpoints',
    name='color_cyclegan_experiment',
    preprocess='none',
    # use 3-channel HSV tensors for new model
    input_nc=3,
    output_nc=3,
    ngf=32,
    ndf=32,
    use_dropout=False,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)


# -----------------------------
# 2. Load model and checkpoint
# -----------------------------
model = ColorCycleGANModel(opt)
checkpoint_path = '../results/checkpoint_epoch_10.pth'
checkpoint = torch.load(checkpoint_path, map_location=opt.device)
try:
    model.netG_A.load_state_dict(checkpoint['netG_A'])
    model.netG_B.load_state_dict(checkpoint['netG_B'])
except Exception as e:
    print(f"Warning: failed to load checkpoint weights cleanly: {e}\nYou may be loading a checkpoint with different input_nc. Proceeding with current weights.")
model.eval()

# -----------------------------
# 3. Load and preprocess image (HSV flow)
# -----------------------------
image_path = './my_datasets/p2.jpg'

img = Image.open(image_path).convert("RGB")
img = img.resize((480,640)) 
img = pad_to_multiple(img, 8)

# convert to HSV tensor in order [V, S, H] with shape [3, H, W]
HSV_tensor = rgb_to_hsv_tensor(img)
HSV_batch = HSV_tensor.unsqueeze(0)
print("Input HSV tensor batch shape:", HSV_batch.shape)

device = opt.device if hasattr(opt, 'device') else next(model.netG_A.parameters()).device
HSV_batch = HSV_batch.to(device)

# -----------------------------
# 4. Transform HSV
# -----------------------------
with torch.no_grad():
    fake_HSV_batch = model.transform_to_analog(HSV_batch)

fake_HSV = fake_HSV_batch[0].cpu()
fake_HSV = torch.clamp(fake_HSV, 0.0, 1.0)
print("Fake HSV tensor shape:", fake_HSV.shape)

# -----------------------------
# 5. Crop to original size and convert back to RGB
# -----------------------------
H, W = img.size[1], img.size[0]
fake_HSV = fake_HSV[:, :H, :W]

rgb_fake = hsv_tensor_to_rgb(fake_HSV)
rgb_fake.show()

