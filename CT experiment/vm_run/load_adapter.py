"""Load an OpenLatent CT adapter and encode CT-FM spatial features.

Input  : 4096-d CT-FM feature vector `s` per scan (last CT-FM feature map, adaptive-avg-pooled to 2x2x2, i.e. 512*8).
Output : 256-d latent z.
Usage  : python load_adapter.py A3_s0_1000steps.safetensors features.npy   (features: N x 4096, float32)
"""
import sys
import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file


def build_encoder(d_in: int = 4096, d_hidden: int = 1024, d_latent: int = 256) -> nn.Module:
    return nn.Sequential(nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_latent)),
                         nn.LayerNorm(d_latent))


def load_adapter(path: str):
    t = load_file(path)
    enc = build_encoder()
    enc.load_state_dict({k[len("encoder."):]: v for k, v in t.items() if k.startswith("encoder.")})
    return enc.eval(), t["input_mean"], t["input_std"]


@torch.no_grad()
def encode(path: str, features: np.ndarray) -> np.ndarray:
    enc, mean, std = load_adapter(path)
    x = (torch.as_tensor(features, dtype=torch.float32) - mean) / std      # same standardisation as training
    return enc(x).numpy()


if __name__ == "__main__":
    z = encode(sys.argv[1], np.load(sys.argv[2]))
    print(z.shape)
