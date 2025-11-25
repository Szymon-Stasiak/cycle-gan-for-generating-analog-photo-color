# models/lut.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

def trilinear(img, lut):
    """
    img: (B, 3, H, W) in [0,1] (expected)
    lut: (3, D, D, D) or (B, 3, D, D, D)
    returns: (B, 3, H, W)
    """
    # normalize to [-1,1] because grid_sample expects normalized coords when align_corners=True
    img = (img - 0.5) * 2.0   # (B,3,H,W)
    # ensure lut has batch dimension
    if lut.ndim == 4:  # (3,D,D,D)
        lut = lut[None]  # (1,3,D,D,D)
    # expand lut to batch size if needed
    if img.shape[0] != lut.shape[0]:
        lut = lut.expand(img.shape[0], -1, -1, -1, -1)  # (B,3,D,D,D)

    # prepare sampling grid: (B, 1, H, W, 3)
    grid = img.permute(0, 2, 3, 1).unsqueeze(1)  # (B,1,H,W,3)

    # grid_sample expects input (B, C, D, H, W) and grid (B, out_D, out_H, out_W, 3)
    # but because lut is (B, 3, D, D, D) we can treat D as depth and sample with grid
    # use mode='bilinear' to get tri-linear interpolation
    out = F.grid_sample(lut, grid, mode='bilinear', padding_mode='border', align_corners=True)
    # out: (B, 3, 1, H, W) -> squeeze dim 2
    out = out.squeeze(2)
    return out


class LUT3D(nn.Module):
    def __init__(self, dim=33, mode='zero', identity_path=None, device=None):
        """
        dim: LUT resolution (e.g., 33 -> 33x33x33)
        mode: 'zero' or 'identity'
        identity_path: optional path to IdentityLUT33.txt or 64 if dim==64
        """
        super().__init__()
        self.dim = dim
        if mode == 'zero':
            lut = torch.zeros(3, dim, dim, dim, dtype=torch.float32)
            self.LUT = nn.Parameter(lut)   # trainable
        elif mode == 'identity':
            # try to load file if provided, else build analytic identity
            if identity_path and os.path.exists(identity_path):
                with open(identity_path, 'r') as f:
                    lines = f.readlines()
                lut = torch.zeros(3, dim, dim, dim, dtype=torch.float32)
                for i in range(dim):
                    for j in range(dim):
                        for k in range(dim):
                            n = i * dim * dim + j * dim + k
                            x = lines[n].split()
                            lut[0, i, j, k] = float(x[0])
                            lut[1, i, j, k] = float(x[1])
                            lut[2, i, j, k] = float(x[2])

                self.LUT = nn.Parameter(lut)
            else:
                # build analytic identity (R/G/B meshgrid flipped to match many LUT formats)
                coords = torch.stack(torch.meshgrid(
                    torch.linspace(0, 1, dim),
                    torch.linspace(0, 1, dim),
                    torch.linspace(0, 1, dim),
                    indexing='ij'
                ), dim=0)  # (3, D, D, D)

                # coords currently R,G,B order; many LUTs expect flipped ordering, but this is fine if consistent.
                self.LUT = nn.Parameter(coords)
        else:
            raise NotImplementedError("mode must be 'zero' or 'identity'")

    def forward(self, img):
        return trilinear(img, self.LUT)
