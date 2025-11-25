import os
import cv2
import torch
import numpy as np
from model.lut import LUT3D  # your lut code

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load LUT
lut = LUT3D(dim=33, mode='identity', identity_path="./IdentityLUT33.txt").to(device)
lut.eval()

def apply_lut_cv2(input_path, output_path):
    # 1. Load with OpenCV → BGR format
    bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)

    if bgr is None:
        print(f"Cannot load {input_path}")
        return

    # 2. Convert BGR → RGB for LUT
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # 3. Convert to float32 tensor in 0–1 range
    rgb_tensor = (
        torch.from_numpy(rgb).float().div(255.0)
        .permute(2, 0, 1)        # HWC → CHW
        .unsqueeze(0)            # batch
        .to(device)
    )

    # 4. Apply LUT
    with torch.no_grad():
        out = lut(rgb_tensor)

    # 5. Convert back to numpy
    out_np = (
        out.squeeze(0)
        .permute(1, 2, 0)        # CHW → HWC
        .cpu().numpy()
    )

    # 6. Convert RGB → BGR
    out_bgr = cv2.cvtColor((out_np * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)

    diff = np.abs(out_bgr - bgr).mean()
    print("Mean pixel difference:", diff)   
    cv2.imshow("image", np.hstack([bgr,out_bgr]))
    cv2.waitKey(0)
    # 7. Save
    cv2.imwrite(output_path, out_bgr)
    print("Saved:", output_path)

apply_lut_cv2("data/tree.jpeg",
                   "data/tree1.jpeg")