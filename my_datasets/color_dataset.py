import os
import glob
from torch.utils.data import Dataset
from PIL import Image
import random
import numpy as np
from utils.color_utils import rgb_to_lab_tensor
import albumentations as A


class UnalignedColorDataset(Dataset):
    """
    Dataset for two domains (unaligned) for CycleGAN color learning.
    Domain A: digital
    Domain B: analog
    """

    def __init__(self, path_A, path_B, transform=None, image_size=256):
        self.dir_A = path_A
        self.dir_B = path_B

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
        A_path = self.A_paths[idx % self.A_size]
        B_path = random.choice(self.B_paths)

        A_img = Image.open(A_path).convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        B_img = Image.open(B_path).convert("RGB").resize((self.size, self.size), Image.BICUBIC)

        L_A, AB_A = rgb_to_lab_tensor(A_img)
        L_B, AB_B = rgb_to_lab_tensor(B_img)

        sample = {
            'L_A': L_A,
            'AB_A': AB_A,
            'L_B': L_B,
            'AB_B': AB_B,
            'path_A': A_path,
            'path_B': B_path
        }

        return sample


class UnalignedColorDatasetAug(Dataset):
    """
    Dataset for CycleGAN learning colors:
    - Each digital and analog photo is used at least once in its original form
    - Additionally, random color variants (augmentation)
    """

    def __init__(self, path_A, path_B, image_size=256):
        self.dir_A = path_A
        self.dir_B = path_B

        print(f"Loading images from {self.dir_A} and {self.dir_B}")

        self.A_paths = sorted(glob.glob(os.path.join(self.dir_A, "*.*")))
        self.B_paths = sorted(glob.glob(os.path.join(self.dir_B, "*.*")))
        print(f"Found {len(self.A_paths)} images in domain A and {len(self.B_paths)} images in domain B.")

        self.size = image_size

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        self.color_aug = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.5),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.5)
        ])
        print(f"Number of images after augmentation: {self.__len__()}")


    def __len__(self):
        return max(self.A_size, self.B_size) * 2

    def __getitem__(self, idx):
        use_aug = idx >= max(self.A_size, self.B_size)
        real_idx = idx % max(self.A_size, self.B_size)

        A_path = self.A_paths[real_idx % self.A_size]
        B_path = random.choice(self.B_paths)

        A_img = Image.open(A_path).convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        B_img = Image.open(B_path).convert("RGB").resize((self.size, self.size), Image.BICUBIC)

        if use_aug:
            A_img_np = np.array(A_img)
            B_img_np = np.array(B_img)
            A_img = self.color_aug(image=A_img_np)['image']
            B_img = self.color_aug(image=B_img_np)['image']
            A_img = Image.fromarray(A_img)
            B_img = Image.fromarray(B_img)

        L_A, AB_A = rgb_to_lab_tensor(A_img)
        L_B, AB_B = rgb_to_lab_tensor(B_img)

        sample = {
            'L_A': L_A,
            'AB_A': AB_A,
            'L_B': L_B,
            'AB_B': AB_B,
            'path_A': A_path,
            'path_B': B_path,
            'augmented': use_aug
        }

        return sample




if __name__ == "__main__":
    dataset = UnalignedColorDataset(root_dir="../data/", image_size=256)
    print(f"Number of images: {len(dataset)}")
    sample = dataset[0]
    print("Shapes:")
    print("L_A:", sample['L_A'].shape)
    print("AB_A:", sample['AB_A'].shape)
    print("L_B:", sample['L_B'].shape)
    print("AB_B:", sample['AB_B'].shape)
