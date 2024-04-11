This repository is for visualizing Muscle Activation from Real-Time SMPL inference from PyMAF repo.

Please follow the next sections to run demo.

# PyMAF [ICCV'21 Oral] & PyMAF-X [TPAMI'23]
This repository contains the code for the following papers:

**PyMAF-X: Towards Well-aligned Full-body Model Regression from Monocular Images**  
Hongwen Zhang, Yating Tian, Yuxiang Zhang, Mengcheng Li, Liang An, Zhenan Sun, Yebin Liu 

TPAMI, 2023

[[Project Page]](https://www.liuyebin.com/pymaf-x) [[Paper]](https://arxiv.org/pdf/2207.06400.pdf) [[Code: smplx branch]](https://github.com/HongwenZhang/PyMAF/tree/smplx)

**PyMAF: 3D Human Pose and Shape Regression with Pyramidal Mesh Alignment Feedback Loop**  
Hongwen Zhang*, Yating Tian*, Xinchi Zhou, Wanli Ouyang, Yebin Liu, Limin Wang, Zhenan Sun 

\* Equal contribution

ICCV, 2021 (Oral Paper)

[[Project Page]](https://hongwenzhang.github.io/pymaf) [[Paper]](https://arxiv.org/pdf/2103.16507.pdf) [Code: smpl branch]

## Requirements

- Python 3.8
```
conda create --no-default-packages -n pymafx python=3.8
conda activate pymafx
```

### packages

- [PyTorch](https://www.pytorch.org) tested on version 1.9.0
```
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c conda-forge

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cudatoolkit=11.3 -c pytorch  # I'm using CUDA 12.4 and this code worked
```

- [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
```
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

- other packages listed in `requirements.txt`
```
pip install -r requirements.txt
```

### necessary files

> mesh_downsampling.npz & DensePose UV data

- Run the following script to fetch mesh_downsampling.npz & DensePose UV data from other repositories.

```
bash fetch_data.sh
```
> SMPL model files

- Collect SMPL model files from [https://smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de) and [UP](https://github.com/classner/up/blob/master/models/3D/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl). Rename model files and put them into the `./data/smpl` directory.

> Fetch preprocessed data from [SPIN](https://github.com/nkolot/SPIN#fetch-data).

> Fetch final_fits data from [SPIN](https://github.com/nkolot/SPIN#final-fits). [important note: using [EFT](https://github.com/facebookresearch/eft) fits for training is much better. Compatible npz files are available [here](https://cloud.tsinghua.edu.cn/d/635c717375664cd6b3f5)]

> Download the [pre-trained model](https://drive.google.com/drive/folders/1R4_Vi4TpCQ26-6_b2PhjTBg-nBxZKjz6?usp=sharing) and put it into the `./data/pretrained_model` directory.

After collecting the above necessary files, the directory structure of `./data` is expected as follows.  
```
./data
├── dataset_extras
│   └── .npz files
├── J_regressor_extra.npy
├── J_regressor_h36m.npy
├── mesh_downsampling.npz
├── pretrained_model
│   └── PyMAF_model_checkpoint.pt
├── smpl
│   ├── SMPL_FEMALE.pkl
│   ├── SMPL_MALE.pkl
│   └── SMPL_NEUTRAL.pkl
├── smpl_mean_params.npz
├── final_fits
│   └── .npy files
└── UV_data
    ├── UV_Processed.mat
    └── UV_symmetry_transforms.mat
```

#### For webcam input:

```
python3 webcam_video.py
```

# pyMAC

This code is an implementation designed to enable the usage of the code for MASS [Lee et al, 2019], Generative GaitNet [Park et al, 2022], and Bidirectional GaitNet [Park et al, 2023] solely through Python libraries without the need for C++.

We checked this code works in Python 3.8, ray(rllib) 2.0.1 and Cluster Server (64 CPUs (128 threads) and 1 GPU (RTX 3090) per node).


## Installation

1. Create Vritual Environment
```bash
python3.8 -m venv pyMAC
source pyMAC/bin/activate
```

2. DartPy installtion from source 

```bash
# Install dependencies following https://dartsim.github.io/install_dartpy_on_ubuntu.html
(pyMAC) git clone https://github.com/dartsim/dart.git
(pyMAC) cd dart
(pyMAC) git checkout tags/v6.11.1
(pyMAC) cp {project_directory}/cpp_files/* python/dartpy/dynamics/
(pyMAC) mkdir build
(pyMAC) cd build
(pyMAC) cmake .. -DCMAKE_BUILD_TYPE=Release -DDART_BUILD_DARTPY=ON
(pyMAC) make -j4 dartpy
(pyMAC) make install-dartpy
```

3. Install other libraries

```bash
(pyMAC) pip3 install --upgrade pip

# Simulation and viwer libraries
(pyMAC) pip3 install PyOpenGL-accelerate PyOpenGL imgui glfw numpy numba gym bvh numpy-quaternion

# DeepRL library 
(pyMAC) pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
## For GaitServer
# pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

(pyMAC) pip3 install ray==2.0.1 
(pyMAC) pip3 install ray[rllib] 
(pyMAC) pip3 install ray[default]

# (Optional) if not working with previous installation
(pyMAC) pip3 install wheel==0.38.4
(pyMAC) pip3 install gym==0.21.0 ## Problemn related to np.bool 
(pyMAC) pip3 install "pydantic<2"
```
## Render 

```bash
cd {project folder}/
(pyMAC) python3 main.py
# (wiht trained_nn) (pyMAC) python3 main.py --checkpoint=trained_nn/lafan_walk_set