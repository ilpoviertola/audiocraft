import typing as tp
import logging
from pathlib import Path

import torch
from torch import nn

from . import vit
from ..utils.jepa.aggregation import (
    ClipAggregation,
    FrameAggregation,
)


logger = logging.getLogger()
# logger.setLevel(logging.INFO)


def load_pretrained(encoder, pretrained, checkpoint_key="target_encoder"):
    logger.info(f"Loading pretrained model from {pretrained}")
    checkpoint = torch.load(pretrained, map_location="cpu")
    try:
        pretrained_dict = checkpoint[checkpoint_key]
    except Exception:
        pretrained_dict = checkpoint["encoder"]

    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {
        k.replace("backbone.", ""): v for k, v in pretrained_dict.items()
    }
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(
                f'key "{k}" is of different shape in model and loaded state dict'
            )
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    # print(encoder)
    logger.info(f"loaded pretrained model with msg: {msg}")
    logger.info(
        f'loaded pretrained encoder from epoch: {checkpoint["epoch"]}\n path: {pretrained}'
    )
    del checkpoint
    return encoder


def init_model(
    device,
    pretrained,
    model_name,
    patch_size,
    crop_size,
    # Video specific parameters
    frames_per_clip,
    tubelet_size,
    use_sdpa,
    use_SiLU,
    tight_SiLU,
    uniform_power,
    checkpoint_key,
):
    encoder = vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=frames_per_clip,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
    )

    encoder.to(device)
    encoder = load_pretrained(
        encoder=encoder, pretrained=pretrained, checkpoint_key=checkpoint_key
    )
    return encoder


def factory(
    device,
    model_name,
    patch_size,
    pretrain_folder,
    checkpoint_name,
    resolution,
    tag=None,
    use_pos_embed=False,
    max_frames=1000,
    use_sdpa=True,
    use_SiLU=False,
    tight_SiLU=True,
    uniform_power=False,
    tubelet_size=2,
    pretrain_frames_per_clip=1,
    checkpoint_key="target_encoder",
    attend_across_segments=False,
    use_spatial_aggregation=True,
    spatial_agg_type="attention",
):
    pretrained_path = f"{pretrain_folder}/{checkpoint_name}"
    # TODO: Add support for downloading on-the-fly
    assert Path(
        pretrained_path
    ).exists(), f"Pretrained model not found at {pretrained_path}"
    encoder = init_model(
        crop_size=resolution,
        device=device,
        pretrained=pretrained_path,
        model_name=model_name,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        frames_per_clip=pretrain_frames_per_clip,
        uniform_power=uniform_power,
        checkpoint_key=checkpoint_key,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
        use_sdpa=use_sdpa,
    )
    # aggreagtion is used if we are processing long videos
    # TODO: Implement dataloading in 16 frame patches and use this aggregation
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    if pretrain_frames_per_clip == 1:
        # Process each frame independently and aggregate
        encoder = FrameAggregation(
            encoder, use_pos_embed=use_pos_embed, max_frames=max_frames
        ).to(device)
    else:
        # Process each video clip independently and aggregate
        encoder = ClipAggregation(
            encoder,
            tubelet_size=tubelet_size,
            attend_across_segments=attend_across_segments,
            use_spatial_aggregation=use_spatial_aggregation,
            spatial_agg_type=spatial_agg_type,
            use_pos_embed=use_pos_embed,
            max_frames=max_frames,
        ).to(device)

    return encoder


class JEPAEncoderWrapper(nn.Module):
    def __init__(
        self,
        model_name: str,
        patch_size: int,
        pretrain_folder: str,
        checkpoint_name: str,
        device: str,
        resolution: int,
        attend_across_segments: bool = False,
        tag: tp.Union[str, None] = None,
        use_pos_embed: bool = False,
        max_frames: int = 1000,
        use_sdpa: bool = True,
        use_silu: bool = False,
        tight_silu: bool = True,
        uniform_power: bool = False,
        tubelet_size: int = 2,
        pretrain_frames_per_clip: int = 1,
        checkpoint_key: str = "target_encoder",
        use_spatial_aggregation: bool = True,
        spatial_agg_type: str = "attention",
    ):
        super().__init__()
        self.model = factory(
            resolution=resolution,
            checkpoint_key=checkpoint_key,
            device=torch.device(device),
            model_name=model_name,
            patch_size=patch_size,
            pretrain_folder=pretrain_folder,
            checkpoint_name=checkpoint_name,
            tag=tag,
            use_pos_embed=use_pos_embed,
            max_frames=max_frames,
            use_sdpa=use_sdpa,
            use_SiLU=use_silu,
            tight_SiLU=tight_silu,
            uniform_power=uniform_power,
            tubelet_size=tubelet_size,
            pretrain_frames_per_clip=pretrain_frames_per_clip,
            attend_across_segments=attend_across_segments,
            use_spatial_aggregation=use_spatial_aggregation,
            spatial_agg_type=spatial_agg_type,
        )

    def forward(self, x, clip_indices=None):
        x = self.model(x, clip_indices)
        return x
