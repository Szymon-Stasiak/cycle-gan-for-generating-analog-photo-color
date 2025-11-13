import cv2
import torch
import numpy as np
from PIL import Image

def rgb_to_lab_tensor(img: Image.Image):
    """
    Convert PIL RGB image to LAB tensors.
    Returns L_tensor (1,H,W) and AB_tensor (2,H,W) normalized to [-1,1].
    """
    lab = cv2.cvtColor(np.array(img).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[:, :, 0]
    A = lab[:, :, 1] - 128.0
    B = lab[:, :, 2] - 128.0

    L_tensor = torch.from_numpy((L / 50.0) - 1.0).unsqueeze(0).float()
    AB_tensor = torch.from_numpy(np.stack([A / 128.0, B / 128.0], axis=0)).float()

    return L_tensor, AB_tensor

def lab_tensor_to_rgb(L_tensor, AB_tensor):
    """
    Convert LAB tensors back to RGB PIL image.
    L_tensor: 1xHxW, AB_tensor: 2xHxW
    """
    L = ((L_tensor.squeeze(0).numpy() + 1.0) * 50.0).astype(np.float32)
    A = (AB_tensor[0].numpy() * 128.0 + 128.0).astype(np.float32)
    B = (AB_tensor[1].numpy() * 128.0 + 128.0).astype(np.float32)

    lab = np.stack([L, A, B], axis=2).astype(np.uint8)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb)


if __name__ == "__main__":
    img = Image.open("../data/1000000544.jpg").convert("RGB")
    L_tensor, AB_tensor = rgb_to_lab_tensor(img)
    print("L tensor shape:", L_tensor.shape)
    print("AB tensor shape:", AB_tensor.shape)

    rgb_converted = lab_tensor_to_rgb(L_tensor, AB_tensor)
    rgb_converted.show()



