import argparse
import os
import shutil
import tempfile
import glob
import sys
import zipfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from my_datasets.color_dataset import UnalignedColorDataset
from my_datasets.paired_color_dataset import PairedColorDataset
from model.color_cyclegan_model import ColorCycleGANModel
from .replay_buffer import ImageBuffer


def find_image_dirs(root):
    """Return directories under root that contain image files, sorted by count desc."""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp")
    counts = {}
    for dirpath, dirs, files in os.walk(root):
        cnt = 0
        for e in exts:
            cnt += len(glob.glob(os.path.join(dirpath, e)))
        if cnt > 0:
            counts[dirpath] = cnt
    # sort by count desc
    sorted_dirs = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [p for p, c in sorted_dirs]


def prepare_filmset(dataset_name, out_dir):
    """Download (via kagglehub) and prepare dataset so that out_dir contains trainA/ and trainB/.
    Heuristics:
      - If extracted dir already contains trainA/trainB, use them
      - Else pick top-2 image-containing subdirs as domain A and B
    Returns the path used as root_dir (out_dir)
    """
    try:
        import kagglehub
    except Exception as e:
        raise RuntimeError("kagglehub is required to download FilmSet. Install it with `pip install kagglehub` or provide dataset manually.") from e

    download_path = kagglehub.dataset_download(dataset_name)
    print("Downloaded dataset to:", download_path)

    # If it's a file (zip), extract it to tempdir
    work_dir = None
    if os.path.isfile(download_path):
        work_dir = tempfile.mkdtemp(prefix="filmset_")
        print("Extracting archive to:", work_dir)
        try:
            shutil.unpack_archive(download_path, work_dir)
        except Exception:
            # fallback to zipfile
            with zipfile.ZipFile(download_path, 'r') as zf:
                zf.extractall(work_dir)
    else:
        # it's a directory
        work_dir = download_path

    # find candidate image dirs
    img_dirs = find_image_dirs(work_dir)
    if not img_dirs:
        raise RuntimeError("No image directories found inside the downloaded dataset.")

    # if trainA/trainB exist, prefer them
    trainA = None
    trainB = None
    candidate = Path(work_dir)
    if (candidate / 'trainA').exists() and (candidate / 'trainB').exists():
        trainA = str(candidate / 'trainA')
        trainB = str(candidate / 'trainB')
    else:
        # pick top 2 dirs by image counts
        if len(img_dirs) >= 2:
            trainA, trainB = img_dirs[0], img_dirs[1]
        elif len(img_dirs) == 1:
            # one directory only: split into two halves
            trainA = img_dirs[0]
            trainB = img_dirs[0]

    # create out_dir/trainA and trainB as symlinks where possible
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    destA = os.path.join(out_dir, 'trainA')
    destB = os.path.join(out_dir, 'trainB')

    def link_or_copy(src, dst):
        if os.path.exists(dst):
            return
        try:
            os.symlink(os.path.abspath(src), dst)
        except Exception:
            # fallback: create directory and copy a tiny index (do not copy full dataset)
            os.makedirs(dst, exist_ok=True)
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp'):
                for f in glob.glob(os.path.join(src, ext)):
                    # create a small text file pointing to original (to avoid copying big files)
                    linkfile = os.path.join(dst, os.path.basename(f) + '.link')
                    with open(linkfile, 'w') as fh:
                        fh.write(f)

    link_or_copy(trainA, destA)
    link_or_copy(trainB, destB)

    print(f"Prepared dataset root at {out_dir} (trainA <- {trainA}, trainB <- {trainB})")
    return out_dir


def build_and_train(root_dir, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # enforce HSV channels
    args.input_nc = 3
    args.output_nc = 3

    if getattr(args, 'paired', False):
        dataset = PairedColorDataset(root_dir=root_dir, image_size=args.image_size)
    else:
        dataset = UnalignedColorDataset(root_dir=root_dir, image_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = ColorCycleGANModel(args)
    model.netG_A.to(device)
    model.netG_B.to(device)
    model.netD_A.to(device)
    model.netD_B.to(device)

    optimizer_G = torch.optim.Adam(
        list(model.netG_A.parameters()) + list(model.netG_B.parameters()),
        lr=args.lr, betas=(0.5, 0.999)
    )
    optimizer_D = torch.optim.Adam(
        list(model.netD_A.parameters()) + list(model.netD_B.parameters()),
        lr=args.lr, betas=(0.5, 0.999)
    )

    fake_A_buffer = ImageBuffer(max_size=50)
    fake_B_buffer = ImageBuffer(max_size=50)

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        for i, data in enumerate(dataloader):
            model.set_input(data)

            # Forward
            model.forward(model.AB_A, model.AB_B)

            # Generators
            optimizer_G.zero_grad()
            model.backward_G(model.AB_A, model.AB_B)
            optimizer_G.step()

            # Discriminators
            optimizer_D.zero_grad()
            model.backward_D(model.AB_A, model.AB_B, fake_A_buffer=fake_A_buffer, fake_B_buffer=fake_B_buffer)
            optimizer_D.step()

            if i % 10 == 0:
                print(f"[Epoch {epoch+1}/{args.epochs}] [Batch {i}/{len(dataloader)}] "
                      f"Loss_G: {model.loss_G.item():.4f} Loss_D_A: {model.loss_D_A.item():.4f} Loss_D_B: {model.loss_D_B.item():.4f}")

        # Save checkpoint
        ckpt = {
            'netG_A': model.netG_A.state_dict(),
            'netG_B': model.netG_B.state_dict(),
            'netD_A': model.netD_A.state_dict(),
            'netD_B': model.netD_B.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'optimizer_D': optimizer_D.state_dict()
        }
        torch.save(ckpt, os.path.join(args.save_dir, f'{args.name}_checkpoint_epoch_{epoch+1}.pth'))

    print("Training finished")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true', help='Download FilmSet via kagglehub')
    parser.add_argument('--dataset_name', type=str, default='xuhangc/filmset', help='kagglehub dataset id')
    parser.add_argument('--dataroot', type=str, default='./data/filmset', help='local dataset root to prepare')
    parser.add_argument('--paired', action='store_true', help='Use paired dataset (input->styled)')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lambda_cycle', type=float, default=24.0)
    parser.add_argument('--lambda_identity', type=float, default=64.0)
    parser.add_argument('--save_dir', type=str, default='../results', help='folder to save results and checkpoints')
    parser.add_argument('--name', type=str, default='filmset_experiment')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='directory for model checkpoints')
    parser.add_argument('--preprocess', type=str, default='none', help='image preprocess mode')
    parser.add_argument('--init_type', type=str, default='normal', help='network weight initialization')
    parser.add_argument('--init_gain', type=float, default=0.02, help='weight initialization gain')
    parser.add_argument('--continue_train', action='store_true', help='continue training from checkpoints')
    parser.add_argument('--load_iter', type=int, default=0, help='which iteration to load (if any)')
    parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load / save')
    parser.add_argument('--verbose', action='store_true', help='print network architectures')
    parser.add_argument('--lr_policy', type=str, default='step', help='learning rate scheduler policy')
    parser.add_argument('--norm', type=str, default='batch', help='normalization type (batch|instance)')
    parser.add_argument('--ngf', type=int, default=32)
    parser.add_argument('--ndf', type=int, default=32)
    parser.add_argument('--use_dropout', action='store_true')

    args = parser.parse_args()

    # Required by BaseModel
    args.isTrain = True

    if args.download:
        root = prepare_filmset(args.dataset_name, args.dataroot)
    else:
        root = args.dataroot

    # convert save_dir relative to repository layout
    # if user supplied '../results' keep as-is
    build_and_train(root, args)


if __name__ == '__main__':
    main()
