import importlib

import torch
import torchvision

torchvision.disable_beta_transforms_warning()
import torchvision.transforms as Tv
from torchvision.transforms import v2


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_video_transforms(transforms_config: list) -> torch.nn.Sequential:
    """Returns a torch.nn.Sequential of video transforms according to the config.

    Args:
        transforms_config (list): Config for the transforms.

    Returns:
        torch.nn.Sequential: Transformations to be applied to the video.
    """
    transforms = []
    for transform_config in transforms_config:
        transform = instantiate_from_config(transform_config)
        transforms.append(transform)
    return torch.nn.Sequential(*transforms)


def get_s3d_transforms() -> Tv.Compose:
    return Tv.Compose(
        [
            Tv.Resize(256, antialias=True),
            Tv.CenterCrop((224, 224)),
            Tv.RandomHorizontalFlip(p=0.5),
            Tv.ConvertImageDtype(torch.float32),
        ]
    )


def get_s3d_transforms_validation() -> Tv.Compose:
    return Tv.Compose(
        [
            Tv.Resize(256, antialias=True),
            Tv.CenterCrop((224, 224)),
            Tv.ConvertImageDtype(torch.float32),
        ]
    )


class ToFloat32DType(torch.nn.Module):
    def __init__(self):
        super().__init__()
        try:
            self.transform = v2.ConvertDtype(torch.float32)
        except AttributeError:
            self.transform = v2.ToDtype(torch.float32, scale=True)

    def forward(self, x):
        return self.transform(x)


class RandomNullify(torch.nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        if torch.rand(1) < self.p:
            return x * 0
        else:
            return x


class Permute(torch.nn.Module):
    def __init__(self, permutation: list):
        super().__init__()
        self.permutation = permutation

    def forward(self, x):
        return x.permute(*self.permutation)
