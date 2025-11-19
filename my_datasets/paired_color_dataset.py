import os
import glob
from torch.utils.data import Dataset
from PIL import Image
from utils.color_utils import rgb_to_hsv_tensor


class PairedColorDataset(Dataset):
    """Dataset for paired images: each input image has a corresponding styled image.

    Expects a root directory containing two subfolders. By default looks for
    'input' and 'styled', falling back to 'trainA' and 'trainB'. Filenames are
    matched by basename intersection.
    """

    def __init__(self, root_dir, input_dir_names=None, image_size=256):
        if input_dir_names is None:
            input_dir_names = ('input', 'styled')

        self.root = root_dir
        # find subdirs
        dirs = os.listdir(root_dir)
        dset_dirs = {d.lower(): os.path.join(root_dir, d) for d in dirs if os.path.isdir(os.path.join(root_dir, d))}

        # choose names
        in_name = None
        out_name = None
        for cand in (input_dir_names + ('trainA', 'trainB')):
            if cand in dset_dirs and in_name is None:
                in_name = dset_dirs[cand]
            elif cand in dset_dirs and out_name is None:
                out_name = dset_dirs[cand]

        # fallback: if only two dirs present, pick them
        if in_name is None or out_name is None:
            subdirs = list(dset_dirs.values())
            if len(subdirs) >= 2:
                in_name, out_name = subdirs[0], subdirs[1]
            elif len(subdirs) == 1:
                in_name = out_name = subdirs[0]
            else:
                raise RuntimeError(f"No suitable subdirectories found in {root_dir}")

        self.dir_A = in_name
        self.dir_B = out_name

        exts = ('*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp')
        def list_images(d):
            files = []
            for e in exts:
                files.extend(glob.glob(os.path.join(d, e)))
            return {os.path.basename(f): f for f in files}

        A_files = list_images(self.dir_A)
        B_files = list_images(self.dir_B)

        # intersection of basenames
        common = sorted(set(A_files.keys()).intersection(set(B_files.keys())))
        if not common:
            # if no exact basenames match, try prefix matching by filename without suffix
            A_basenames = {os.path.splitext(k)[0]: v for k, v in A_files.items()}
            B_basenames = {os.path.splitext(k)[0]: v for k, v in B_files.items()}
            common_keys = sorted(set(A_basenames.keys()).intersection(set(B_basenames.keys())))
            common = [k + os.path.splitext(list(A_files.keys())[0])[1] for k in common_keys] if common_keys else []

        # build paired lists
        self.pairs = []
        for name in common:
            a = A_files.get(name)
            b = B_files.get(name)
            if a and b:
                self.pairs.append((a, b))

        # if still empty, try zip ordering (best-effort)
        if not self.pairs:
            A_list = sorted(A_files.values())
            B_list = sorted(B_files.values())
            L = min(len(A_list), len(B_list))
            for i in range(L):
                self.pairs.append((A_list[i], B_list[i]))

        self.size = image_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        A_path, B_path = self.pairs[idx]
        A_img = Image.open(A_path).convert('RGB').resize((self.size, self.size), Image.BICUBIC)
        B_img = Image.open(B_path).convert('RGB').resize((self.size, self.size), Image.BICUBIC)

        HSV_A = rgb_to_hsv_tensor(A_img)
        HSV_B = rgb_to_hsv_tensor(B_img)

        sample = {
            'V_A': HSV_A[0:1, :, :],
            'S_A': HSV_A[1:2, :, :],
            'H_A': HSV_A[2:3, :, :],
            'V_B': HSV_B[0:1, :, :],
            'S_B': HSV_B[1:2, :, :],
            'H_B': HSV_B[2:3, :, :],
            'A_path': A_path,
            'B_path': B_path
        }

        return sample


if __name__ == '__main__':
    ds = PairedColorDataset(root_dir='.', image_size=128)
    print(len(ds))
    s = ds[0]
    print(s['V_A'].shape, s['V_B'].shape)
