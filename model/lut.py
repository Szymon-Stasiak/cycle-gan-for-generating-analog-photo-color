# models/lut.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
import cv2

def trilinear(img, lut):

    """
    img: (B, 3, H, W) in [0,1] (expected)
    lut: (3, D, D, D) or (B, 3, D, D, D)
    returns: (B, 3, H, W)
    """

    img = (img - 0.5) * 2.0
    if lut.ndim == 4:
        lut = lut[None]
    if img.shape[0] != lut.shape[0]:
        lut = lut.expand(img.shape[0], -1, -1, -1, -1)
    grid = img.permute(0, 2, 3, 1).unsqueeze(1)
    out = F.grid_sample(lut, grid, mode='bilinear', padding_mode='border', align_corners=True)
    out = out.squeeze(2)
    return out


class LUT3D(nn.Module):
    def __init__(self, dim=33, mode='zero', path=None):
        
        """
        dim: LUT resolution (e.g., 33 -> 33x33x33)
        mode: 'zero' or 'txtLoad'
        path: optional path to LUT txt file or 64 if dim==64
        """
        
        super().__init__()
        self.dim = dim
        if mode == 'zero':
            lut = torch.zeros(3, dim, dim, dim, dtype=torch.float32)
            self.LUT = nn.Parameter(lut)
        elif mode == 'txtLoad':
            if path and os.path.exists(path):
                with open(path, 'r') as f:
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
                raise NotImplementedError("Provide the correct path to the LUT txt file")
        else:
            raise NotImplementedError("mode must be 'zero' or 'txtLoad'")

    def forward(self, img):
        return trilinear(img, self.LUT)

class LUTDataset(Dataset):
    def __init__(self, trainA, trainB):

        """
        trainA: path to starting image
        trainB: path to processed image
        """

        self.trainA = trainA
        self.trainB = trainB
        self.files = sorted(os.listdir(trainA))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        A = cv2.imread(os.path.join(self.trainA, name))[:, :, ::-1] / 255.0
        B = cv2.imread(os.path.join(self.trainB, name))[:, :, ::-1] / 255.0

        A  = torch.from_numpy(A ).permute(2,0,1).float()
        B  = torch.from_numpy(B ).permute(2,0,1).float()

        return A, B

def train_lut(trainA, trainB, dim=33, epochs=200, lr=0.01, batch=1, workers=1):
    
    """
    trainA: path to starting image
    trainB: path to processed image
    dim: LUT resolution (e.g., 33 -> 33x33x33)
    epochs: amount of complete pass through the entire training set
    lr: learning rate to pass to ADAM (Adaptive Moment Estimation) optimizer
    batch: size of each batch to be processed at the same time
    workers: how many samples to be loaded concurrently
    returns: trained LUT for that training dataset
    """

    dataset = LUTDataset(trainA, trainB)
    loader  = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=workers)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lut = LUT3D(dim=dim, mode='zero').to(device)
    adam = optim.Adam([lut.LUT], lr=lr)
    for e in range(epochs):
        total_loss = 0
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = lut(x)
            loss = torch.mean((pred-y)**2)
            adam.zero_grad()
            loss.backward()
            adam.step()
            total_loss += loss.item()
        print(f"[{e+1}/{epochs}] Loss = {total_loss/len(loader):.6f}")
    return lut

def image_resizer(folders=[], size=512):

    """
    folders: path to folders that contain the images you plan to resize
    size: size you want the images to be (e.g., 512 -> 512x512)
    """

    for fold in folders:
        files = os.listdir(fold)
        counter = 0
        for f in files:
            if(counter%100 == 0): print(counter)
            counter += 1
            path = f"{fold}/{f}"
            img = cv2.imread(path)
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            cv2.imwrite(path, img)
        print("Folder complete")
    print("Images resized")