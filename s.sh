#!/bin/bash
#SBATCH --job-name="Imagenet_training"
#SBATCH --partition=A100-80GB,A100-PCI,A100-40GB
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=150G
#SBATCH --time=72:00:00

export PATH=$HOME/.local/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=OFF
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

srun \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
  --container-mounts=/ds:/ds:ro,"`pwd`":"`pwd`" \
  --container-workdir="`pwd`" \
  bash -c "python3 -m pip install --upgrade timm pip torchvision scikit-learn && \
           torchrun --standalone  --nproc_per_node=4 project8.py"
