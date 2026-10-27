#!/usr/bin/env bash
# Prepare a Lambda (or Colab) GPU VM for the privacy-obfuscation experiments.
# Idempotent: safe to rerun after a reboot or a partial run.
#
#   - system packages: ffmpeg (frame extraction), unrar/p7zip (HMDB51 archives), tmux
#   - python packages on top of the image's torch: video/image backbones, autoencoders
#   - this repo at ~/openlatent (branch privacy-research)
#
# Usage: bash setup_vm.sh
set -euo pipefail

BRANCH="${BRANCH:-privacy-research}"
REPO=https://github.com/AcastaPaloma/openlatent

if ! command -v ffmpeg >/dev/null || ! command -v unrar >/dev/null; then
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg unrar p7zip-full tmux rsync >/dev/null
fi

python3 -m pip install -q --upgrade pip
# video decoding goes through the ffmpeg CLI, so no PyAV/decord (no py3.10 wheels for recent PyAV)
python3 -m pip install -q timm==1.0.15 diffusers==0.33.1 \
  transformers==4.51.3 accelerate==1.6.0 huggingface_hub==0.30.2 einops==0.8.1 \
  pandas scikit-learn tensorboard opencv-python-headless==4.11.0.86 pyyaml

if [ ! -d ~/openlatent/.git ]; then
  git clone -q -b "$BRANCH" "$REPO" ~/openlatent
else
  git -C ~/openlatent fetch -q origin && git -C ~/openlatent checkout -q "$BRANCH" \
    && git -C ~/openlatent pull -q --ff-only origin "$BRANCH"
fi

mkdir -p ~/work/{data,runs,logs}
python3 -c "import torch, torchvision, timm, diffusers; print('torch', torch.__version__, 'tv', torchvision.__version__, 'cuda', torch.cuda.is_available())"
echo "setup done: $(hostname)"
