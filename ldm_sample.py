# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torchmetrics
import torch
from diffusers import SD3Transformer2DModel
from diffusers.models import AutoencoderKL
from torch.utils.data import Dataset
from tqdm import tqdm

from src.dataset.fastmri import utils
from src.dataset.fastmri.data.subsample import create_mask_for_mask_type
from src.dataset.fastmri.data.transforms import to_tensor, complex_to_chan_dim, apply_mask, chan_complex_to_last_dim
from src.module.sampler import create_sampler_solver
from src.module.sampler.measurement import MRISystemMatrix

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from src.module.models.diffusion import create_diffusion
import argparse

def normalize_torch(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img -= torch.amin(img, dim=(1, 2, 3), keepdim=True)  # np.min(img)
    img /= torch.amax(img, dim=(1, 2, 3), keepdim=True)  # np.max(img)
    return img

def intersect_dicts(da, db, exclude=()):
    """Returns a dictionary of intersecting keys with matching shapes, excluding 'exclude' keys, using da values."""
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}

def load_pretrained_parameters(model, pretrained, verbose=True):
    ckpt = torch.load(pretrained, map_location='cpu')
    # if 'ema' in ckpt:
    #     ckpt = ckpt['ema']
    if 'model' in ckpt:
        ckpt = ckpt['model']
    print(f"Loading pretrained model parameters from '{pretrained}'. ")
    # csd = csd_copy  # checkpoint state_dict
    cmsd = model.state_dict()  # current model state_dict with required grad
    csd = intersect_dicts(ckpt, cmsd)  # intersect
    unmatching_keys = [k for k, v in cmsd.items() if k not in csd.keys()] if len(csd) != len(cmsd) else []
    model.load_state_dict(csd, strict=False)  # load
    if verbose:
        print(f'Transferred {len(csd)}/{len(cmsd)} items from pretrained weights, unmatching keys are:{unmatching_keys}')
    return model


class MRIVolumeDataset(Dataset):
    def __init__(self, root, start_slice=0, num_slices=0):
        self.root = Path(root)
        self.subject_dir = sorted(list(self.root.glob("*/")))
        self.start_slice = start_slice
        self.num_slices = num_slices

    def __len__(self):
        return len(self.subject_dir)

    def normalize(self, img):
        '''
            Estimate mvue from coils and normalize with 99% percentile.
        '''
        scaling = torch.quantile(img.abs(), 0.99)
        return img / scaling

    def __getitem__(self, idx):
        subject_dir = self.subject_dir[idx]
        img_path = subject_dir / "slice"
        # mps_path = subject_dir / "mps"
        img_files = sorted(list(img_path.glob("*.npy")))
        if self.num_slices != 0:
            img_files = img_files[self.start_slice: self.start_slice + self.num_slices]
        else:
            img_files = img_files[self.start_slice:]
        # load data
        imgs = []
        mps_list = []
        filenames = []
        for img_file in img_files:
            img = to_tensor(np.load(img_file)).float()[None,...]
            img = self.normalize(img)
            mps_file = subject_dir / "mps" / img_file.name
            mps = to_tensor(np.load(mps_file)).float()
            imgs.append(img)
            mps_list.append(mps)
            filenames.append(img_file.stem)
        imgs = torch.stack(imgs) # [B, 1, H, W, 2]
        mps = torch.stack(mps_list) # [B, Coils, H, W, 2]

        return imgs, mps, filenames


def create_dataloader(args):
    from torch.utils.data import DataLoader

    dataset = MRIVolumeDataset(
        root=args.data_path,
        start_slice=args.start_slice,
        num_slices=args.num_slices,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=getattr(dataset, 'collate_fn', None),
        persistent_workers=True,
    )
    return dataloader


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # save path
    save_path = Path(args.save) / 'inference' / f"AF-X{args.acceleration}"
    save_path.mkdir(parents=True, exist_ok=True)

    measurement_path = save_path / "measurement"
    sample_path = save_path / "sample"
    reference_path = save_path / "reference"
    measurement_path.mkdir(parents=True, exist_ok=True)
    reference_path.mkdir(parents=True, exist_ok=True)
    sample_path.mkdir(parents=True, exist_ok=True)

    # Initial model:
    latent_size = args.image_size // 8
    vae = AutoencoderKL.from_pretrained(args.vae, subfolder="vae").to(device)
    vae.requires_grad_(False)

    model_kwargs = {
        "sample_size": 40,
        "patch_size": 2,
        "in_channels": 16,
        "num_layers": 18,
        "attention_head_dim": 64,
        "num_attention_heads": 12,
        "joint_attention_dim": 1024,
        "caption_projection_dim": 768,
        "pooled_projection_dim": 1024,
        "out_channels": 32,  # learn sigma
        "pos_embed_max_size": 192,
    }
    model = SD3Transformer2DModel(**model_kwargs).to(device)

    # Load model checkpoint:
    ckpt_path = args.ckpt
    model = load_pretrained_parameters(model, ckpt_path)
    model.eval()  # important!
    solver = create_sampler_solver(str(args.num_sampling_steps))

    # Load data:
    dataloader = create_dataloader(args)

    # Set up metrics
    psnr = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0)
    lpips = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity()
    ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0)
    # save metrics
    columns = ['FileName', 'PSNR', 'SSIM', 'LPIPS']
    all_df = pd.DataFrame(columns=columns)
    df_output_file = save_path / 'metrics.csv'
    if not Path(df_output_file).exists():
        all_df.to_csv(df_output_file, index=False)
    # initialize the mask function
    mask_func = create_mask_for_mask_type(
        args.mask_type, args.center_fraction, args.acceleration,
        random_acc=False
    )

    # Diffusion solver:
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        img, mps, fname = batch
        img, mps = img[0], mps[0]
        fname = [f[0] for f in fname]
        mps = mps.to(device)
        # prepare the input
        _, mask, _ = apply_mask(utils.fft2c(img), mask_func, seed=args.seed, padding=(0, 320))

        img = complex_to_chan_dim(img)
        img = img.to(device)
        mask = mask.to(device)

        # Prepare operator function
        operator = MRISystemMatrix(mask, mps)
        y = operator.AT(operator.A(img))  # measurement

        samples = solver.solve(model, vae, y, operator, latent_size, progress=True)
        samples = samples[:, :2, ...]

        # Post-process
        samples = chan_complex_to_last_dim(samples)
        samples = utils.complex_abs(samples)
        masked_image = chan_complex_to_last_dim(y)
        masked_image = utils.complex_abs(masked_image)
        target = chan_complex_to_last_dim(img)
        target = utils.complex_abs(target)
        # normalize
        masked_image = normalize_torch(masked_image)
        samples = normalize_torch(samples)
        target = normalize_torch(target)

        # Save results and display images:
        batch_size = target.size(0)
        for idx in range(batch_size):
            # display images
            save_image(masked_image[idx:idx + 1], measurement_path / f"{fname[idx]}_measurement.png", nrow=1, normalize=True, value_range=(0, 1))
            save_image(samples[idx:idx + 1], sample_path / f"{fname[idx]}_sample.png", nrow=1, normalize=True, value_range=(0, 1))
            save_image(target[idx:idx + 1], reference_path / f"{fname[idx]}_reference.png", nrow=1, normalize=True, value_range=(0, 1))
            # compute metrics
            recon_image = samples[idx: idx + 1].detach().cpu()
            target_image = target[idx: idx + 1].cpu()
            psnr_value = psnr(recon_image, target_image).item()
            ssim_value = ssim(recon_image, target_image).item()
            recon_image = torch.repeat_interleave(recon_image, 3, dim=1) * 2 - 1
            target_image = torch.repeat_interleave(target_image, 3, dim=1) * 2 - 1
            lpips_value = lpips(
                torch.clip(recon_image, min=-1, max=1),
                torch.clip(target_image, min=-1, max=1)
            ).item()
            # append to csv
            new_row = pd.DataFrame([[fname[idx], psnr_value, ssim_value, lpips_value]], columns=columns)
            new_row.to_csv(df_output_file, mode='a', header=False, index=False)

        # # TODO: delete after testing
        if i >= 0:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=r"./brain_multicoil/multicoil_test_processed/", help="Path to the dataset.")
    parser.add_argument("--start-slice", type=int, default=0, help="start slice of per volume.")
    parser.add_argument("--num-slices", type=int, default=1, help="number slices of per volume, '0' means all slices.")
    parser.add_argument("--save", type=str, default="./runs", help="Path for saving running related data.")
    parser.add_argument("--vae", default='./model_weight/stable-diffusion-3-medium-diffusers', type=str, help="vae path")
    parser.add_argument("--num-workers", type=int, default=1, help="number of dataloader process.")
    parser.add_argument("--mask-type", type=str, default="equispaced_fraction")
    parser.add_argument("--acceleration", type=int, default=4, help="acceleration factor")
    parser.add_argument("--center-fraction", type=float, default=0.08, help="ACS region")
    parser.add_argument("--image-size", default=320, type=int, choices=[320])
    parser.add_argument("--num-sampling-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default='./runs/train_ldm/brain/checkpoints/model_ema.pt', help="Optional path to a model checkpoint.")
    args = parser.parse_args()
    main(args)
