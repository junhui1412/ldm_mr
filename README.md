# A Detail-preserving Latent Diffusion Model for Arbitrarily Accelerated MR Imaging


## 💻 Local Setup

### 1. Prepare the environment


- python 3.10  
- PyTorch 2.5.0  
- CUDA 12.1  

Other versions of PyTorch with proper CUDA should work but are not fully tested.

```bash
# in CGRS folder
conda create -n ldm_mr python=3.10
conda activate ldm_mr

# install PyTorch with proper CUDA
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.1 -c pytorch -c nvidia
# or
# pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

### 2. Prepare the dataset

- Install BART for sensitivity map estimation

1. ```sudo apt-get install make gcc libfftw3-dev liblapacke-dev libpng-dev libopenblas-dev```
1. ```wget https://github.com/mrirecon/bart/archive/v0.6.00.tar.gz```
1. ```tar xzvf v0.6.00.tar.gz```
1. ```cd bart-0.6.00```
1. ```make```

- Download data

Download fastMRI dataset from the [fastMRI dataset page](https://fastmri.med.nyu.edu/).

- Script for estimating sensitivity maps from data
You can found the script `estimate_maps.py` in the `preprocess` folder, and you can run it to estimate the sensitivity maps from the fastMRI data.
```bash
python estimate_maps.py --input-dir=./brain_multicoil/multicoil_train/ --output-dir=./brain_multicoil/multicoil_train_processed/ --organ=brain
```

### 3. Train your model
At first, you need to download the pretrained `vae` model from the [stabilityai](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers/tree/main) and
put it into the `./model_weight/stable-diffusion-3-medium-diffusers/vae/` folder.

Then, you can modify the config file `./cfg/train_configs/train_ldm.yaml` to set your training parameters.

Now you are ready to run. 

```
python train_ldm.py --config=./cfg/train_configs/train_ldm.yaml
```

### 4. Inference

You can modify the parameters in the following command to run inference.
```bash
python ldm_sample.py --data-path ./brain_multicoil/multicoil_test/ --num-slices 0 \
--save ./runs --acceleration 8 --center-fraction 0.04 --num-sampling-steps 1000 \
--ckpt /your/trained/model/path/model.pt
```
