import os
import glob
from torch.utils.data import Dataset
from PIL import Image
import random
from utils.color_utils import rgb_to_lab_tensor

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


if __name__ == "__main__":
    dataset = UnalignedColorDataset(root_dir="../data/", image_size=256)
    print(f"Number of images: {len(dataset)}")
    sample = dataset[0]
    print("Shapes:")
    print("L_A:", sample['L_A'].shape)
    print("AB_A:", sample['AB_A'].shape)
    print("L_B:", sample['L_B'].shape)
    print("AB_B:", sample['AB_B'].shape)
