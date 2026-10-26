"""Compatibility shim for running upstream code on current torch/torchvision.

pytorchvideo 0.1.3 imports torchvision.transforms.functional_tensor, which torchvision
removed in 0.17. The only use is AugMix (augmentations.py), and only its `affine`
function, which torchvision.transforms.functional provides. This project never uses
AugMix, so aliasing the module is enough for the import chain to succeed.

Import this module before anything that imports pytorchvideo.
"""

import sys

import torchvision.transforms.functional as _tv_functional

sys.modules.setdefault("torchvision.transforms.functional_tensor", _tv_functional)
