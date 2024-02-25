"""Miscellaneous utility functions."""
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from threading import Thread
from collections import OrderedDict


########## PyTorch ##########


def copy_tensor(x, grad=False):
    """Copy a tensor."""
    return x.clone().detach().requires_grad_(grad)


def torch2numpy(x):
    """Convert a Tensor to a numpy array."""
    if isinstance(x, float):
        return x
    return x.detach().cpu().numpy()


def norm_batch(x):
    """Compute norm with batch-axis left."""
    return x.view(x.shape[0], -1).norm(p=2, dim=1)


def torch2image(x, data_range="[-1,1]"):
    """Convert torch tensor in [-1, 1] scale to be numpy array format
    image in (N, H, W, C) in [0, 255] scale.
    """
    if data_range == "[-1,1]":
        x = (x.clamp(-1, 1) + 1) / 2
    x = (x * 255).cpu().numpy()
    if len(x.shape) == 4:
        x = x.transpose(0, 2, 3, 1)
    elif len(x.shape) == 3:
        x = x.transpose(1, 2, 0)  # (C, H, W)
    return x.astype("uint8")


def norm_img(x):
    """Normalize an arbitrary Tensor to [-1, 1]"""
    return (x - x.min()) / (x.max() - x.min()) * 2 - 1


def image2torch(x):
    """Process [0, 255] (N, H, W, C) numpy array format
    image into [0, 1] scale (N, C, H, W) torch tensor.
    """
    y = torch.from_numpy(x).float() / 255.0
    if len(x.shape) == 3 and x.shape[2] == 3:
        return y.permute(2, 0, 1).unsqueeze(0)
    if len(x.shape) == 4:
        return y.permute(0, 3, 1, 2)
    return 0


def pil2torch(img):
    return image2torch(np.asarray(img))


def bu(img, size, align_corners=True):
    """Bilinear interpolation with Pytorch.

    Args:
      img : a list of tensors or a tensor.
    """
    if isinstance(img, list):
        return [
            F.interpolate(i, size=size, mode="bilinear", align_corners=align_corners)
            for i in img
        ]
    return F.interpolate(img, size=size, mode="bilinear", align_corners=align_corners)


def tocpu(x):
    """Convert to CPU Tensor."""
    return x.clone().detach().cpu()


def preprocess_image(arr):
    """Preprocess np.array into tensor image.
    Args:
        arr: [H, W, 3] in [0, 255]
    Returns:
        [1, 3, H, W] in [-1, 1]
    """
    x = torch.from_numpy(arr).unsqueeze(0).float() / 255.
    return x.permute(0, 3, 1, 2) * 2 - 1


def imread_tensor(fpath):
    """Read to 0-1 tensor in [3, H, W]."""
    x = torch.from_numpy(imread(fpath))
    return x.permute(2, 0, 1).float() / 255.


def imread_pil(fpath, size=None):
    img = Image.open(open(fpath, "rb")).convert("RGB")
    if size is not None:
        img = img.resize(size, resample=Image.Resampling.BILINEAR)
    return img


def clip_sample(x):
    """Clip the sampling result."""
    return x.clamp(0, 1)


def D(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return x


def set_params_grad(params, grad=False):
    for param in params:
        param.requires_grad = grad

    
########## Other ###########


class DictRecorder(OrderedDict):
    def __init__(self):
        super().__init__()

    def add(self, key, val):
        if key not in self:
            self[key] = []
        if isinstance(val, torch.Tensor):
            self[key].append(D(val))
        elif isinstance(val, list):
            for i, v in enumerate(val):
                if len(self[key]) <= i:
                    self[key].append([])
                self[key][i].append(D(v))
        else:
            self[key].append(val)
    
    def add_dict(self, dic):
        for key, val in dic.items():
            self.add(key, val)

    def stack(self):
        for key in list(self.keys()):
            if len(self[key]) > 0:
                if isinstance(self[key][0], torch.Tensor):
                    self[key] = torch.stack(self[key]).squeeze()
                elif isinstance(self[key][0], list):
                    dic = {}
                    for i in range(len(self[key])):
                        dic[str(i)] = torch.stack(self[key][i])
                    self[key] = dic
            else:
                del key


def dict_append(dic, val, key1, key2=None, key3=None):
    """Create a list or append to it with at most 3 levels."""
    if key1 and not key2 and not key3:
        if key1 not in dic:
            dic[key1] = []
        dic[key1].append(val)
        return

    if key1 not in dic:
        dic[key1] = {}
    if key1 and key2 and not key3:
        if key2 not in dic:
            dic[key1][key2] = []
        dic[key1][key2].append(val)
        return

    if key2 not in dic[key1]:
        dic[key1][key2] = {}
    if key3 not in dic[key1][key2]:
        dic[key1][key2][key3] = []
    dic[key1][key2][key3].append(val)


def imwrite(fpath, arr):
    with open(fpath, "wb") as f:
        Image.fromarray(arr).save(f)


def imread(fpath):
    img = Image.open(open(fpath, "rb")).convert("RGB")
    return np.asarray(img).copy()


class GeneralThread(Thread):
    """Function interface threading."""

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args, self.kwargs = args, kwargs

    def run(self):
        self.res = self.func(*self.args, **self.kwargs)
