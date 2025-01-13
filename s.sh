#!/bin/bash
#SBATCH --job-name="imagenet_training"
#SBATCH --partition=batch
#SBATCH --gpus=3
#SBATCH --cpus-per-gpu=50
#SBATCH --mem=250G
#SBATCH --time=24:00:00

srun \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
  --container-mounts=/ds:/ds:ro,"`pwd`":"`pwd`" \
  --container-workdir="`pwd`" \
  bash -c "python3 -m pip install timm scikit-learn && python3 project4.py"
