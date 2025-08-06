"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import math
from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from .. import utils


from .subsample import MaskFunc


def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.

    For complex arrays, the real and imaginary parts are stacked along the last
    dimension.

    Args:
        data: Input numpy array.

    Returns:
        PyTorch version of dataset.
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)


def tensor_to_complex_np(data: torch.Tensor) -> np.ndarray:
    """
    Converts a complex torch tensor to numpy array.

    Args:
        data: Input dataset to be converted to numpy.

    Returns:
        Complex numpy version of dataset.
    """
    return torch.view_as_complex(data).numpy()

def complex_to_chan_dim(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w, two = x.shape
    assert two == 2
    return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

def chan_complex_to_last_dim(x: torch.Tensor) -> torch.Tensor:
    b, c2, h, w = x.shape
    assert c2 % 2 == 0
    c = c2 // 2
    return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

def apply_mask(
    data: torch.Tensor,
    mask_func: MaskFunc,
    offset: Optional[int] = None,
    seed: Optional[Union[int, Tuple[int, ...]]] = None,
    padding: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data: The input k-space dataset. This should have at least 3 dimensions,
            where dimensions -3 and -2 are the spatial dimensions, and the
            final dimension has size 2 (for complex values).
        mask_func: A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed: Seed for the random number generator.
        padding: Padding value to apply for mask.

    Returns:
        tuple containing:
            masked dataset: Subsampled k-space dataset.
            mask: The generated mask.
            num_low_frequencies: The number of low-resolution frequency samples
                in the mask.
    """
    shape = (1,) * len(data.shape[:-3]) + tuple(data.shape[-3:])
    mask, num_low_frequencies = mask_func(shape, offset, seed)
    if padding is not None:
        mask[..., : padding[0], :] = 0
        mask[..., padding[1] :, :] = 0  # padding value inclusive on right of zeros

    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

    return masked_data, mask, num_low_frequencies


def mask_center(x: torch.Tensor, mask_from: int, mask_to: int) -> torch.Tensor:
    """
    Initializes a mask with the center filled in.

    Args:
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.

    Returns:
        A mask with the center filled.
    """
    mask = torch.zeros_like(x)
    mask[:, :, :, mask_from:mask_to] = x[:, :, :, mask_from:mask_to]

    return mask

def get_pad_and_num_low_freqs(
    mask: torch.Tensor, num_low_frequencies: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    if num_low_frequencies is None or num_low_frequencies == 0:
        # get low frequency line locations and mask them out
        squeezed_mask = mask[:, 0, 0, :, 0].to(torch.int8)
        cent = squeezed_mask.shape[1] // 2
        # running argmin returns the first non-zero
        left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
        right = torch.argmin(squeezed_mask[:, cent:], dim=1)
        num_low_frequencies_tensor = torch.max(
            2 * torch.min(left, right), torch.ones_like(left)
        )  # force a symmetric center unless 1
    else:
        num_low_frequencies_tensor = num_low_frequencies * torch.ones(
            mask.shape[0], dtype=mask.dtype, device=mask.device
        )

    pad = (mask.shape[-2] - num_low_frequencies_tensor + 1) // 2

    return pad.type(torch.long), num_low_frequencies_tensor.type(torch.long)

def batched_mask_center(
    x: torch.Tensor, mask_from: torch.Tensor, mask_to: torch.Tensor
) -> torch.Tensor:
    """
    Initializes a mask with the center filled in.

    Can operate with different masks for each batch element.

    Args:
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.

    Returns:
        A mask with the center filled.
    """
    if not mask_from.shape == mask_to.shape:
        raise ValueError("mask_from and mask_to must match shapes.")
    if not mask_from.ndim == 1:
        raise ValueError("mask_from and mask_to must have 1 dimension.")
    if not mask_from.shape[0] == 1:
        if (not x.shape[0] == mask_from.shape[0]) or (
            not x.shape[0] == mask_to.shape[0]
        ):
            raise ValueError("mask_from and mask_to must have batch_size length.")

    if mask_from.shape[0] == 1:
        mask = mask_center(x, int(mask_from), int(mask_to))
    else:
        mask = torch.zeros_like(x)
        for i, (start, end) in enumerate(zip(mask_from, mask_to)):
            mask[i, :, :, start:end] = x[i, :, :, start:end]

    return mask


def center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data: The input tensor to be center cropped. It should
            have at least 2 dimensions and the cropping is applied along the
            last two dimensions.
        shape: The output shape. The shape should be smaller
            than the corresponding dimensions of dataset.

    Returns:
        The center cropped image.
    """
    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        raise ValueError("Invalid shapes.")

    h_from = (data.shape[-2] - shape[0]) // 2
    w_from = (data.shape[-1] - shape[1]) // 2
    h_to = h_from + shape[0]
    w_to = w_from + shape[1]

    return data[..., h_from:h_to, w_from:w_to]


def complex_center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input image or batch of complex images.

    Args:
        data: The complex input tensor to be center cropped. It should have at
            least 3 dimensions and the cropping is applied along dimensions -3
            and -2 and the last dimensions should have a size of 2.
        shape: The output shape. The shape should be smaller than the
            corresponding dimensions of dataset.

    Returns:
        The center cropped image
    """
    # if not (0 < shape[0] <= dataset.shape[-3] and 0 < shape[1] <= dataset.shape[-2]):
    #     raise ValueError("Invalid shapes.")


    # w_from = (dataset.shape[-3] - shape[0]) // 2
    # h_from = (dataset.shape[-2] - shape[1]) // 2
    # w_to = w_from + shape[0]
    # h_to = h_from + shape[1]

    # return dataset[..., w_from:w_to, h_from:h_to, :]

    if not (0 < shape[0] <= data.shape[-3] and 0 < shape[1] <= data.shape[-2]):
        raise ValueError("Invalid shapes.")

    h_from = (data.shape[-3] - shape[0]) // 2
    w_from = (data.shape[-2] - shape[1]) // 2
    h_to = h_from + shape[0]
    w_to = w_from + shape[1]

    return data[..., h_from:h_to, w_from:w_to, :]


def center_crop_to_smallest(
    x: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply a center crop on the larger image to the size of the smaller.

    The minimum is taken over dim=-1 and dim=-2. If x is smaller than y at
    dim=-1 and y is smaller than x at dim=-2, then the returned dimension will
    be a mixture of the two.

    Args:
        x: The first image.
        y: The second image.

    Returns:
        tuple of tensors x and y, each cropped to the minimim size.
    """
    smallest_width = min(x.shape[-1], y.shape[-1])
    smallest_height = min(x.shape[-2], y.shape[-2])
    x = center_crop(x, (smallest_height, smallest_width))
    y = center_crop(y, (smallest_height, smallest_width))

    return x, y

def complex_pad(x: torch.Tensor, img_sz, mode='replicate'):
    """Batch complex pad."""
    if x.shape[-1] != 2 and x.shape[-1] != 1:
        raise ValueError("Last dimension must be 2 for complex or 1 for masks.")
    n_dim = len(x.shape)
    if n_dim == 4:
        x = x.unsqueeze(0)
    x = x.permute(0, 1, 4, 2, 3) # (b, c, h, w, 2) -> (b, c, 2, h, w)
    h_1, w_1 = x.shape[-2:]
    h_2, w_2 = img_sz
    h_pad = [math.floor((h_2 - h_1) / 2), math.ceil((h_2 - h_1) / 2)]
    w_pad = [math.floor((w_2 - w_1) / 2), math.ceil((w_2 - w_1) / 2)]
    c_pad = [0, 0]
    x = F.pad(x, w_pad + h_pad + c_pad, mode=mode).permute(0, 1, 3, 4, 2) # (b, c, 2, h, w) -> (b, c, h, w, 2)
    if n_dim == 4:
        return x.squeeze(0)
    return x

def pad(x: torch.Tensor, img_sz, mode='replicate') -> Tuple[torch.Tensor, torch.Tensor]:
    h_1, w_1 = x.shape[-2:]
    h_2, w_2 = img_sz
    h_pad = [math.floor((h_2 - h_1) / 2), math.ceil((h_2 - h_1) / 2)]
    w_pad = [math.floor((w_2 - w_1) / 2), math.ceil((w_2 - w_1) / 2)]
    x = F.pad(x, w_pad + h_pad, mode=mode)
    pad_mask = torch.zeros_like(x)
    pad_mask[..., h_pad[0]: h_2 - h_pad[1], w_pad[0]: w_2 - w_pad[1]] = 1
    pad_mask = pad_mask.to(dtype=torch.bool)
    return x, pad_mask

def normalize(
    data: torch.Tensor,
    mean: Union[float, torch.Tensor],
    stddev: Union[float, torch.Tensor],
    eps: Union[float, torch.Tensor] = 0.0,
) -> torch.Tensor:
    """
    Normalize the given tensor.

    Applies the formula (dataset - mean) / (stddev + eps).

    Args:
        data: Input dataset to be normalized.
        mean: Mean value.
        stddev: Standard deviation.
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        Normalized tensor.
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(
    data: torch.Tensor, eps: Union[float, torch.Tensor] = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batch instance normalize.
    Normalize the given tensor with instance norm/

    Applies the formula (dataset - mean) / (stddev + eps), where mean and stddev
    are computed from the dataset itself.

    Args:
        data: Input dataset to be normalized. shape (c, h, w, 2)
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        torch.Tensor: Normalized tensor
    """
    n_dim = len(data.shape)
    if n_dim == 4:
        data = data.unsqueeze(0)

    assert len(data.shape) == 5, f"The shape of dataset is {data.shape}, which is not correct!"

    b, c, h, w, two = data.shape
    data = data.permute(0, 1, 4, 2, 3).view(b, c, two, -1)

    mean = data.mean(dim=-1).view(b, c, two, 1, 1).permute(0, 1, 3, 4, 2)
    std = data.std(dim=-1).view(b, c, two, 1, 1).permute(0, 1, 3, 4, 2)

    data = data.view(b, c, two, h, w).permute(0, 1, 3, 4, 2)

    if n_dim == 4:
        data = data.squeeze(0)
        mean = mean.squeeze(0)
        std = std.squeeze(0)

    return normalize(data, mean, std, eps), mean, std

