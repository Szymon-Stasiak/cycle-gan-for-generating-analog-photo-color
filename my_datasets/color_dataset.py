import os
import glob
from torch.utils.data import Dataset
from PIL import Image
import random
from utils.color_utils import rgb_to_hsv_tensor, rgb_to_lab_tensor

class UnalignedColorDataset(Dataset):
    """
    Dataset for two domains (unaligned) for CycleGAN color learning.
    Domain A: digital
    Domain B: analog
    """

    def __init__(self, root_dir, transform=None, image_size=256):
        """
        root_dir: folder containing subfolders 'trainA' and 'trainB'
        transform: optional augmentations
        image_size: image size after resize
        """
        self.dir_A = os.path.join(root_dir, "trainA")
        self.dir_B = os.path.join(root_dir, "trainB")
        print(f"Loading images from {self.dir_A} and {self.dir_B}")
        self.A_paths = sorted(glob.glob(os.path.join(self.dir_A, "*.*")))
        self.B_paths = sorted(glob.glob(os.path.join(self.dir_B, "*.*")))
        print(f"Found {len(self.A_paths)} images in domain A and {len(self.B_paths)} images in domain B.")
        self.transform = transform
        self.size = image_size

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

    def __len__(self):
        return max(self.A_size, self.B_size)

    def __getitem__(self, idx):
        if len(self.A_paths) >= len(self.B_paths):
            A_path = self.A_paths[idx % self.A_size]
            B_path = self.B_paths[random.randint(0, self.B_size - 1)]
        else:
            A_path = self.A_paths[random.randint(0, self.A_size - 1)]
            B_path = self.B_paths[idx % self.B_size]

        A_img = Image.open(A_path).convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        B_img = Image.open(B_path).convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        
        HSV_A = rgb_to_hsv_tensor(A_img)
        HSV_B = rgb_to_hsv_tensor(B_img)
        sample = {
            'V_A': HSV_A[0:1, :, :],  # Value channel
            'S_A': HSV_A[1:2, :, :],  # Saturation channel
            'H_A': HSV_A[2:3, :, :],  # Hue channel
            'V_B': HSV_B[0:1, :, :],
            'S_B': HSV_B[1:2, :, :],
            'H_B': HSV_B[2:3, :, :],
            'A_path': A_path,
            'B_path': B_path   
        }

        return sample


if __name__ == "__main__":
    dataset = UnalignedColorDataset(root_dir="../data/", image_size=256)
    print(f"Number of images: {len(dataset)}")
    sample = dataset[0]
    print("Shapes:")
    print("V_A:", sample['V_A'].shape)
    print("S_A:", sample['S_A'].shape)
    print("H_A:", sample['H_A'].shape)
    print("V_B:", sample['V_B'].shape)
    print("S_B:", sample['S_B'].shape)
    print("H_B:", sample['H_B'].shape)