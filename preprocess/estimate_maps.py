import sys, os
sys.path.insert(0, '../../programs/bart-0.9.00/python')
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOOLBOX_PATH"]    = '../../programs/bart-0.9.00'
sys.path.append('../../programs/bart-0.9.00/python')

# import glob
import numpy as np
import h5py
import click
import sigpy as sp
from bart import bart
from pathlib import Path
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

def center_crop(data, shape):
    """
    Apply a center crop to the input real image or batch of real images.
    Args:
        data: The input image or batch of images to be cropped.
        shape: The desired output shape (height, width) for the cropped image.
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

def replace_specific_parent(path: Path, old_name: str, new_name: str) -> Path:
    parts = list(path.parts)
    parts = [new_name if part == old_name else part for part in parts]
    return Path(*parts)

# for file in file_list:
def process(file, slice_dir, mps_dir, slice_start, slice_end):
    slice_dir = replace_specific_parent(slice_dir, 'filename_dir', file.stem)
    mps_dir = replace_specific_parent(mps_dir, 'filename_dir', file.stem)
    slice_dir.mkdir(parents=True, exist_ok=True)
    mps_dir.mkdir(parents=True, exist_ok=True)
    filename = file.stem

    # just check the first slice if the output exists or not
    fname = filename + f'_{slice_start:02d}'
    output_slice_name = slice_dir / fname
    output_mps_name = mps_dir / fname
    if output_slice_name.with_suffix('.npy').exists() and output_mps_name.with_suffix('.npy').exists():
        return f'{file} already exists!'

    with h5py.File(file, 'r') as data:
        #num_slices = int(data.attrs['num_slices'])
        kspace = np.array(data['kspace'])
        # s_maps = np.zeros( kspace.shape, dtype = kspace.dtype)
        h, w = kspace.shape[-2:]
        if h < 320 or w < 320: # filter out small images
            return f"{filename} is too small, size: {h}x{w}"
        num_slices = kspace.shape[0]
        num_coils = kspace.shape[1]
        for slice_idx in range(slice_start, num_slices - slice_end):
            fname = filename + f'_{slice_idx:02d}'
            output_slice_name = slice_dir / fname
            output_mps_name = mps_dir / fname
            if output_slice_name.with_suffix('.npy').exists() and output_mps_name.with_suffix('.npy').exists():
                continue

            gt_ksp = kspace[slice_idx]
            print('processing', file, slice_idx)
            try:
                s_maps_ind = bart(1, 'ecalib -m1 -W -c0', gt_ksp.transpose((1, 2, 0))[None,...]).transpose( (3, 1, 2, 0)).squeeze()
            except Exception as e:
                print(f"{file} {slice_idx}: {e}")
                continue
            # s_maps_ind = bart(1, 'ecalib -m1 -W -c0.02', gt_ksp.transpose((1, 2, 0))[None, ...]).transpose((3, 1, 2, 0)).squeeze()
            # s_maps[ slice_idx ] = s_maps_ind
            image = np.sum(sp.ifft(gt_ksp, axes=(-1, -2)) * np.conj(s_maps_ind), axis=0)

            image = center_crop(image, (320, 320))
            s_maps_ind = center_crop(s_maps_ind, (320, 320))

            np.save(output_slice_name, image) # [h, w]
            np.save(output_mps_name, s_maps_ind) # [coil, h, w]
    return "success!"

@click.command()
@click.option('--input-dir', default=None, help='directory with raw data ')
@click.option('--output-dir', default=None, help='output directory for maps')
@click.option('--organ', default='knee', help='"knee" or "brain"')
@click.option('--np', default=1, help='number of processes')
def main(input_dir, output_dir, organ, np):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    slice_dir = output_dir / 'filename_dir' / 'slice'
    mps_dir = output_dir / 'filename_dir' / 'mps'
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        # slice_dir.mkdir(parents=True, exist_ok=True)
        # mps_dir.mkdir(parents=True, exist_ok=True)


# Get all files
# !!! Highly recommended to use 'sorted' otherwise different PCs
# !!! return different orders
#     file_list = sorted(glob.glob(input_dir + '/*.h5'))
    file_list = sorted(list(input_dir.rglob('*.h5')))

    if organ == 'knee':
        slice_start = 5
        slice_end = 0
    elif organ == 'brain':
        slice_start = 0
        slice_end = 2
    else:
        raise ValueError(f"'organ' is {organ}, this type is not supported!")


    processpool = mp.Pool(processes=np)
    main_func = partial(
        process,
        slice_dir=slice_dir,
        mps_dir=mps_dir,
        slice_start=slice_start,
        slice_end=slice_end,
    )
    processes = processpool.imap_unordered(main_func, file_list, chunksize=5)
    for stdout in tqdm(processes, total=len(file_list), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        print(stdout)
        pass

    processpool.close() # close the pool
    processpool.join() # wait for all processes to finish



if __name__ == '__main__':
    main()

