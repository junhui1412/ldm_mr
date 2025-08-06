from collections.abc import Callable
from functools import partial

import torch
from torch import nn

from src.module.models.diffusion.gaussian_diffusion import _extract_into_tensor
from src.module.models.diffusion import SpacedDiffusion


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

class Sampler(SpacedDiffusion):
    def __init__(self, use_timesteps, **kwargs):
        super().__init__(use_timesteps, **kwargs)

    def vae_encode(self, vae, x):
        if vae.config.shift_factor is not None:
            z = (vae.encode(x).latent_dist.sample() - vae.config.shift_factor) * vae.config.scaling_factor
        else:
            z = vae.encode(x).latent_dist.sample() * vae.config.scaling_factor
        return z

    def vae_decode(self, vae, z):
        if vae.config.shift_factor is not None:
            x = vae.decode(z / vae.config.scaling_factor + vae.config.shift_factor).sample
        else:
            x = vae.decode(z / vae.config.scaling_factor).sample
        return x

    def solve(self, model, vae, y, A: Callable, latent_size, eta=1.0, progress=False):
        device = next(model.parameters()).device

        y_3c = torch.cat([y, y.sum(dim=1, keepdim=True) * 0.5], dim=1)  # [*, 3, h, w]
        y_latent = self.vae_encode(vae, y_3c)
        prompt_embeds, pooled_prompt_embeds = encode_prompt(y_latent)

        model_kwargs = {
            "encoder_hidden_states": prompt_embeds,
            "pooled_projections": pooled_prompt_embeds,
            "return_dict": True,
        }
        # init z_T
        zT = torch.randn((y_latent.size(0), y_latent.size(1), latent_size, latent_size)).to(device)

        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        z0_preds = []
        zs = [zT]

        for idx, i in enumerate(indices):

            t = torch.tensor([i] * zT.shape[0], device=device)

            alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, zT.shape)
            alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, zT.shape)

            zt = zs[-1].to(device)

            # 0. NFE
            et = model(hidden_states=zt, timestep=t, **model_kwargs).sample
            et = et[:, :et.size(1) // 2]

            # 1. Tweedie
            z0_t = (zt - et * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
            # 2. pixel update
            x0_t = self.vae_decode(vae, z0_t)
            delta_z0_t = z0_t - self.vae_encode(vae, x0_t)
            x0_t_hat = dps(x0_t[:, :2, ...], A, y) # [*, 2, h, w]
            x0_t_hat = torch.cat([x0_t_hat, x0_t_hat.sum(dim=1, keepdim=True) * 0.5], dim=1)  # [*, 3, h, w]
            z0_t_hat = self.vae_encode(vae, x0_t_hat) + delta_z0_t

            # 3. DDIM sampling
            c1 = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
            c2 = torch.sqrt(1 - alpha_bar_prev - c1 ** 2)
            if idx != len(indices) - 1:
                zt_next = alpha_bar_prev.sqrt() * z0_t_hat + c1 * torch.randn_like(z0_t_hat) + c2 * et
            # Final step
            else:
                zt_next = z0_t_hat

            z0_preds.append(z0_t.to('cpu'))
            zs.append(zt_next.to('cpu'))
        z = zs[-1]
        x = self.vae_decode(vae, z.to(device))
        x = dps(x[:, :2, ...], A, y)
        return x


def compute_residual(x, y, A):
    x = x.reshape(*y.shape)
    return (y - A(x)).norm()

def dps(x0_t, A, y, scaler=1.0):
    Ax_fn = torch.func.jacrev(partial(compute_residual, y=y, A=A))
    norm_grad = Ax_fn(x0_t)
    x0_t = x0_t - norm_grad * scaler
    return x0_t