#!/bin/bash
#SBATCH --job-name="Imagenet_training"
#SBATCH --partition=A100-80GB,A100-PCI,A100-40GB
#SBATCH --gpus=3
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=250G
#SBATCH --time=72:00:00

srun \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
  --container-mounts=/ds:/ds:ro,"`pwd`":"`pwd`" \
  --container-workdir="`pwd`" \
  bash -c "python3 -m pip install --upgrade timm pip torchvision scikit-learn && python3 project7.py"
