import argparse
from copy import deepcopy
import logging
import os

from pathlib import Path
from collections import OrderedDict
import shutil

import numpy as np
import torch
import torch.utils.checkpoint
from diffusers import AutoencoderKL, SD3Transformer2DModel
from einops import rearrange
from omegaconf import OmegaConf
from torch import nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset

from accelerate import Accelerator
# from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from src.dataset.fastmri import utils
from src.dataset.fastmri.data.subsample import create_mask_for_mask_type
from src.dataset.fastmri.data.transforms import to_tensor, apply_mask, complex_to_chan_dim, chan_complex_to_last_dim

import wandb
import math
from torchvision.utils import make_grid

from src.module.models.diffusion import create_diffusion


def normalize_torch(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img -= torch.amin(img, dim=(1, 2, 3), keepdim=True)  # np.min(img)
    img /= torch.amax(img, dim=(1, 2, 3), keepdim=True)  # np.max(img)
    return img

def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    # x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    x = x.mul(255).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x

@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    device = moments.device

    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias)
    return z

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger

def intersect_dicts(da, db, exclude=()):
    """Returns a dictionary of intersecting keys with matching shapes, excluding 'exclude' keys, using da values."""
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}

def load_pretrained_parameters(model, pretrained, logger=None, verbose=True):
    ckpt = torch.load(pretrained, map_location='cpu')
    # if 'ema' in ckpt:
    #     ckpt = ckpt['ema']
    if 'model' in ckpt:
        ckpt = ckpt['model']
    if logger is not None:
        logger.info(f"Loading pretrained model parameters from '{pretrained}'. ")
    # csd = csd_copy  # checkpoint state_dict
    cmsd = model.state_dict()  # current model state_dict with required grad
    csd = intersect_dicts(ckpt, cmsd)  # intersect
    unmatching_keys = [k for k, v in cmsd.items() if k not in csd.keys()] if len(csd) != len(cmsd) else []
    model.load_state_dict(csd, strict=False)  # load
    if verbose and logger is not None:
        logger.info(f'Transferred {len(csd)}/{len(cmsd)} items from pretrained weights, unmatching keys are:{unmatching_keys}')
    return model

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def replace_specific_parent(path: Path, old_name: str, new_name: str) -> Path:
    parts = list(path.parts)
    parts = [new_name if part == old_name else part for part in parts]
    return Path(*parts)

def patchify(imgs, patch_size):
    """
    imgs: (N, C, H, W)
    x: (N, L, patch_size**2 *C)
    """
    p = patch_size
    assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

    c, h, w = imgs.shape[1], imgs.shape[2] // p, imgs.shape[3] // p
    x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
    return x

def encode_prompt(masked_latent):
    # pooling the masked_latent embeddings
    pooled_masked_latent_embeds = nn.AdaptiveAvgPool2d((8, 8))(masked_latent).view(masked_latent.size(0), -1)
    # [b, c, h, w] -> [b, h*w, c]
    masked_latent_embeds = patchify(masked_latent, 8)
    return masked_latent_embeds, pooled_masked_latent_embeds


class MRIDataset(Dataset):
    def __init__(self, root, mask_type, acceleration, center_fraction, random_acc, start_slice=0, num_slices=0, use_seed=True):
        self.root = Path(root)
        self.subject_dir = sorted(list(self.root.glob("*/")))
        self.slice_file = []
        for subject in self.subject_dir:
            img_path = subject / "slice"
            img_files = sorted(list(img_path.glob("*.npy")))
            if num_slices != 0:
                self.slice_file.extend(img_files[start_slice: start_slice + num_slices])
            else:
                self.slice_file.extend(img_files[start_slice:])

        self.mask_func = create_mask_for_mask_type(mask_type, center_fraction, acceleration, random_acc)
        self.use_seed = use_seed

    def __len__(self):
        return len(self.slice_file)

    def normalize(self, img):
        '''
            Estimate mvue from coils and normalize with 99% percentile.
        '''
        scaling = torch.quantile(img.abs(), 0.99)
        return img / scaling

    def __getitem__(self, idx):
        img_path = self.slice_file[idx]
        mps_path = replace_specific_parent(img_path, "slice", "mps")
        img = to_tensor(np.load(img_path)).float()[None, ...] # [1, H, W, 2]
        img = self.normalize(img)
        mps = to_tensor(np.load(mps_path)).float() # [Coils, H, W, 2]
        mc_img = utils.complex_mul(mps, img)  # img [Coils, H, W, 2]
        mc_ksp = utils.fft2c(mc_img)  # kspace [Coils, H, W, 2]

        filename = img_path.name
        seed = None if not self.use_seed else tuple(map(ord, filename))
        acq_start = 0
        acq_end = img.shape[-2]

        masked_kspace, _, _ = apply_mask(
            mc_ksp, self.mask_func, seed=seed, padding=(acq_start, acq_end)
        )

        masked_img = utils.complex_mul(utils.complex_conj(mps), utils.ifft2c(masked_kspace)).sum(dim=0, keepdim=True)

        return img, masked_img, filename


def create_dataloader(args, accelerator, split='train', logger=None):

    if split == 'train':
        dataset = MRIDataset(
            root=args.train_data_path,
            mask_type=args.mask_type,
            acceleration=args.acceleration,
            center_fraction=args.center_fraction,
            random_acc=args.random_acc,
            start_slice=args.start_slice,
            num_slices=args.num_slices,
        )
    elif split == 'val': # val
        dataset = MRIDataset(
            root=args.val_data_path,
            mask_type=args.mask_type,
            acceleration=args.acceleration,
            center_fraction=args.center_fraction,
            random_acc=args.random_acc,
            start_slice=args.start_slice,
            num_slices=args.num_slices,
        )
    else:
        raise ValueError(f'Unknown split {split}')

    if accelerator.is_main_process and logger is not None:
        logger.info(f"{split} Dataset contains {len(dataset):,} images ({getattr(args, f'{split}_data_path')})")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=getattr(dataset, 'collate_fn', None),
        persistent_workers=True,
    )
    return dataloader


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main():
    args = OmegaConf.load(parse_args())
    # set accelerator
    logging_dir = Path(args.output_dir, args.project_name, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=str(Path(args.output_dir, args.project_name)), logging_dir=str(logging_dir)
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        # cpu=True, # for debugging.
    )

    save_dir = Path(args.output_dir, args.project_name, args.exp_name)
    checkpoint_dir = save_dir / "checkpoints"  # Stores saved model checkpoints
    args.save_dir = str(save_dir)
    if accelerator.is_main_process:
        save_dir.mkdir(mode=0o777, parents=True, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        # Save to a yaml file
        OmegaConf.save(args, save_dir / 'args.yaml')
        checkpoint_dir.mkdir(mode=0o777, parents=True, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    else:
        logger = None
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # Create model:
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8

    vae = AutoencoderKL.from_pretrained(args.vae_path, subfolder="vae").to(device)
    requires_grad(vae, False)

    model_kwargs = {
            "sample_size": 40, # 40*40*16
            "patch_size": 2,
            "in_channels": 16,
            "num_layers": 18,
            "attention_head_dim": 64, # 64*24 = 1536
            "num_attention_heads": 12, # 24
            "joint_attention_dim": 1024, # 4096
            "caption_projection_dim": 768, # 1536
            "pooled_projection_dim": 1024, # 2048
            "out_channels": 32, # learn sigma
            "pos_embed_max_size": 192,
    }
    model = SD3Transformer2DModel(**model_kwargs)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule

    # load pre-trained model
    if args.pretrained is not None:
        model = load_pretrained_parameters(model, args.pretrained, logger)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora dit)
    # to half-precision as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    model = model.to(device).to(weight_dtype)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)

    if accelerator.is_main_process:
        logger.info(f"Latent Diffusion Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer and learning rate scheduler:
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Setup mri data:
    train_dataloader = create_dataloader(args, accelerator, split='train', logger=logger)
    val_dataloader = create_dataloader(args, accelerator, split='val', logger=logger)

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    global_step = 0
    first_epoch = 0

    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        tracker_config = OmegaConf.to_container(args, resolve=True)
        accelerator.init_trackers(
            project_name=args.project_name,
            config=tracker_config,
            init_kwargs={
                "wandb": {
                    "name": f"{args.exp_name}",
                    "dir": save_dir,
                }
            },
        )

    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    if accelerator.is_main_process:
        logger.info("***** Running training *****")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(checkpoint_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(str(checkpoint_dir / str(path)))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Create sampling noise (feel free to change):
    total_batch_size = 64 if 16 * accelerator.num_processes >= 64 else 16 * accelerator.num_processes
    sample_batch_size = total_batch_size // accelerator.num_processes
    xT = torch.randn((sample_batch_size, vae.config.latent_channels, latent_size, latent_size), device=device)

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            # data preparation
            images, masked_images, filenames = batch
            images = images.to(device)
            masked_images = masked_images.to(device)
            images = complex_to_chan_dim(images)  # [*, 1, h, w, 2] -> [*, 2, h, w]
            images = torch.cat([images, images.sum(dim=1, keepdim=True) * 0.5], dim=1)  # [*, 3, h, w]
            masked_images = complex_to_chan_dim(masked_images)  # [*, 1, h, w, 2] -> [*, 2, h, w]
            masked_images = torch.cat([masked_images, masked_images.sum(dim=1, keepdim=True) * 0.5], dim=1)  # [*, 3, h, w]
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                if vae.config.shift_factor is not None:
                    latent = (vae.encode(images).latent_dist.sample() - vae.config.shift_factor) * vae.config.scaling_factor
                    masked_latent = (vae.encode(masked_images).latent_dist.sample() - vae.config.shift_factor) * vae.config.scaling_factor
                else:
                    latent = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
                    masked_latent = vae.encode(masked_images).latent_dist.sample() * vae.config.scaling_factor
            model.train()

            with accelerator.accumulate(model):
                t = torch.randint(0, diffusion.num_timesteps, (latent.shape[0],), device=device)
                prompt_embeds, pooled_prompt_embeds = encode_prompt(masked_latent)
                model_kwargs = {
                    "encoder_hidden_states": prompt_embeds,
                    "pooled_projections": pooled_prompt_embeds,
                    "return_dict": True,
                }
                loss_dict = diffusion.training_losses(model, latent, t, model_kwargs)
                loss = loss_dict["loss"].mean()

                ## optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(ema, model)  # change ema function

            # calculate gpu memory usage
            mem = f'{torch.cuda.memory_reserved() / 2 ** 30 if torch.cuda.is_available() else 0.0:.3g}G'
            ### enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0 and global_step > 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(checkpoint_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = checkpoint_dir / removing_checkpoint
                                    shutil.rmtree(removing_checkpoint)

                        checkpoint_path = checkpoint_dir / f"checkpoint-{global_step}"
                        accelerator.save_state(str(checkpoint_path))
                        checkpoint = {
                            "model": accelerator.unwrap_model(model).state_dict(),
                            "ema": ema.state_dict(),
                            "opt": optimizer.state_dict(),
                            "steps": global_step,
                        }
                        ema_checkpoint_path = f"{checkpoint_dir}/model_ema.pt"
                        torch.save(checkpoint, ema_checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}, ema to {ema_checkpoint_path}")

                if (global_step == 1 or (global_step % args.sample_every == 0 and global_step > 0)):
                    for val_step, val_batch in enumerate(val_dataloader):
                        # data preparation
                        images, masked_images, filenames = val_batch
                        images = images.to(device)
                        masked_images = masked_images.to(device)
                        # images = complex_to_chan_dim(images)  # [*, 1, h, w, 2] -> [*, 2, h, w]
                        masked_images = complex_to_chan_dim(masked_images)  # [*, 1, h, w, 2] -> [*, 2, h, w]
                        masked_images = torch.cat([masked_images, masked_images.sum(dim=1, keepdim=True) * 0.5], dim=1)  # [*, 3, h, w]
                        with torch.no_grad():
                            # Map input images to latent space + normalize latents:
                            if vae.config.shift_factor is not None:
                                masked_latent = (vae.encode(masked_images).latent_dist.sample() - vae.config.shift_factor) * vae.config.scaling_factor
                            else:
                                masked_latent = vae.encode(masked_images).latent_dist.sample() * vae.config.scaling_factor
                        with torch.no_grad():
                            val_diffusion = create_diffusion(timestep_respacing=str(args.sample_steps))
                            prompt_embeds, pooled_prompt_embeds = encode_prompt(masked_latent)
                            model_kwargs = {
                                "encoder_hidden_states": prompt_embeds,
                                "pooled_projections": pooled_prompt_embeds,
                                "return_dict": True,
                            }
                            latent_samples = val_diffusion.ddim_sample_loop(accelerator.unwrap_model(model.eval()), xT.shape, noise=xT, clip_denoised=False, model_kwargs=model_kwargs, device=device).to(torch.float32)

                            latent_samples = latent_samples[:, :vae.config.latent_channels, ...]
                            if vae.config.shift_factor is not None:
                                samples = vae.decode(latent_samples / vae.config.scaling_factor + vae.config.shift_factor).sample
                            else:
                                samples = vae.decode(latent_samples / vae.config.scaling_factor).sample
                            # drop last channel
                            samples = samples[:, :2, ...]

                            samples = chan_complex_to_last_dim(samples)
                            samples = utils.complex_abs(samples)
                        out_samples = accelerator.gather(samples.to(torch.float32))
                        target_images = accelerator.gather(utils.complex_abs(images.to(torch.float32)))
                        accelerator.log(
                            {
                                "samples": wandb.Image(array2grid(normalize_torch(out_samples))),
                            },
                            step=global_step,
                        )
                        accelerator.log(
                            {
                                "targets": wandb.Image(array2grid(normalize_torch(target_images))),
                            },
                            step=global_step,
                        )
                        logger.info("Generating EMA samples done.")
                        if val_step > 0: # sample a batch per GPUs
                            break

            logs = {
                "loss": accelerator.gather(loss).mean().detach().item(),
                "gpu_memory": mem,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        # default="cfg/train_configs/train_ldm.yaml",
        required=True,
        help="path to config",
    )
    args = parser.parse_args()
    return args.config


if __name__ == "__main__":
    # for wandb logging
    # os.environ["WANDB_API_KEY"] = 'api_key'
    # os.environ["WANDB_MODE"] = 'dryrun'
    main()
