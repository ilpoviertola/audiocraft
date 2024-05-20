from dataclasses import dataclass, fields, replace
import json
from pathlib import Path
import random
import typing as tp
from fractions import Fraction
import logging

import numpy as np
import torch
from torchvision.io import read_video

from .info_audio_dataset import (
    InfoAudioDataset,
)
from ..modules.conditioners import (
    SegmentWithAttributes,
    ConditioningAttributes,
    VideoCondition,
    WavCondition,
)
from ..utils.transforms import get_video_transforms


EPS = torch.finfo(torch.float32).eps
logger = logging.getLogger(__name__)


@dataclass
class VideoInfo(SegmentWithAttributes):
    video: tp.Optional[tp.Union[torch.Tensor, tp.List[tp.List[torch.Tensor]]]] = None
    self_wav: tp.Optional[torch.Tensor] = None

    @property
    def has_video(self):
        return self.video is not None

    def to_condition_attributes(self) -> ConditioningAttributes:
        out = ConditioningAttributes()

        for _field in fields(self):
            key, value = _field.name, getattr(self, _field.name)
            if key == "self_wav":
                out.wav[key] = value
            else:
                out.video[key] = value
        return out

    @staticmethod
    def attribute_getter(attribute):
        preprocess_func = None
        if attribute in ["self_wav", "video"]:
            preprocess_func = None
        return preprocess_func

    @classmethod
    def from_dict(cls, dictionary: dict, fields_required: bool = False):
        _dictionary: tp.Dict[str, tp.Any] = {}

        # allow a subset of attributes to not be loaded from the dictionary
        # these attributes may be populated later
        post_init_attributes = ["self_wav", "video"]

        for _field in fields(cls):
            if _field.name in post_init_attributes:
                continue
            elif _field.name not in dictionary:
                if fields_required:
                    raise KeyError(f"Unexpected missing key: {_field.name}")
            else:
                preprocess_func: tp.Optional[tp.Callable] = cls.attribute_getter(
                    _field.name
                )
                value = dictionary[_field.name]
                if preprocess_func:
                    value = preprocess_func(value)
                _dictionary[_field.name] = value
        return cls(**_dictionary)


class VideoDataset(InfoAudioDataset):
    # TODO: this is now just copy of SoundDataset
    """Sound audio dataset: Audio dataset with environmental sound-specific metadata.

    Args:
        info_fields_required (bool): Whether all the mandatory metadata fields should be in the loaded metadata.
        aug_p (float): Probability of performing audio mixing augmentation on the batch.
        kwargs: Additional arguments for AudioDataset.

    See `audiocraft.data.info_audio_dataset.InfoAudioDataset` for full initialization arguments.
    """

    def __init__(
        self,
        *args,
        info_fields_required: bool = True,
        aug_p: float = 0.0,
        clip_video: bool = True,
        frames_per_clip: int = 16,
        max_frames: int = 64,
        frame_step: int = 1,
        video_transforms: tp.Optional[tp.List[tp.Dict[str, tp.Any]]] = None,
        **kwargs,
    ):
        assert kwargs["segment_duration"] > 0, "Segment duration must be positive."
        kwargs["return_info"] = (
            True  # We require the info for each song of the dataset.
        )
        super().__init__(*args, **kwargs)
        self.info_fields_required = info_fields_required
        self.aug_p = aug_p
        self.clip_video = clip_video
        self.frames_per_clip = frames_per_clip
        self.max_frames = max_frames
        self.num_clips = max_frames // frames_per_clip
        self.frame_step = frame_step

        self.video_transforms = (
            get_video_transforms(video_transforms) if video_transforms else None
        )

    def _split_video_into_clips(self, video: torch.Tensor):
        """Split video into a list of clips"""
        fpc = self.frames_per_clip
        nc = self.num_clips
        # add one nested list for different spatial views
        # not really supported at the moment but V-JEPA expects this
        # possible e.g. add different crops from same clip (same temporally but spatially different)
        return [[video[:, i * fpc : (i + 1) * fpc]] for i in range(nc)]

    @staticmethod
    def _read_video(
        fn: str,
        start_pts: tp.Optional[tp.Union[float, Fraction]] = 0,
        end_pts: tp.Optional[tp.Union[float, Fraction]] = None,
        pts_unit: str = "sec",
        output_format: str = "TCHW",
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, dict]:
        """Read video file to frames and audio streams.

        Args:
            fn (str): Path to video file.
            start_pts (Optional[Union[float, Fraction]], optional): Where to start reading video from. Defaults to 0.
            end_pts (Optional[Union[float, Fraction]], optional): Where to end reading video to. Defaults to None.
            pts_unit (str, optional): Unit of measurement. Defaults to "sec".
            output_format (str, optional): Output dim order. Defaults to "TCHW".

        Raises:
            FileNotFoundError: If video does not exist.

        Returns:
            (Tensor, Tensor, dict): Frames (Tv, C, H, W), audio streams (Ta, ), and metadata.
        """
        if not Path(fn).is_file():
            raise FileNotFoundError(f"File {fn} does not exist.")

        frames, audio, metadata = read_video(
            fn, start_pts, end_pts, pts_unit, output_format
        )
        audio = audio.mean(dim=0)  # (2, T) -> (T,)
        return frames, audio, metadata

    def __getitem__(self, index):
        # sometimes loading these videos is pain in the ass due to the missing v/a frames...
        loaded_video = False
        wav, info = super().__getitem__(index)
        while not loaded_video:
            video_path = Path(info.meta.path).with_suffix(".mp4")
            video_path = video_path.resolve().as_posix()
            # try to read a EPS seconds more video and then crop to desired sample lens
            try:
                rgb, _, video_meta = self._read_video(
                    video_path,
                    start_pts=info.seek_time,
                    # TODO: check is this duration the whole clip?
                    end_pts=info.seek_time + self.segment_duration + EPS,
                )
                loaded_video = (
                    "video_fps" in video_meta
                    and rgb.shape[0] >= video_meta["video_fps"] * self.segment_duration
                )  # missing frames?
            except FileNotFoundError as e:
                logger.error("File not found %s", str(e))
                loaded_video = False
            if not loaded_video:
                logger.debug(
                    "Could not load video %s. Skipping...",
                    video_path,
                )
                index = np.random.randint(self.__len__())
                wav, info = super().__getitem__(index)

        info_data = info.to_dict()
        video_info = VideoInfo.from_dict(
            info_data, fields_required=self.info_fields_required
        )

        rgb = rgb[: self.max_frames]
        rgb_nframes = rgb.shape[0]
        rgb = self.video_transforms(rgb) if self.video_transforms else rgb
        # rgb = [self._split_video_into_clips(rgb)] if self.clip_video else rgb[None]
        clip_indices = (
            _get_clip_indices(
                rgb_nframes,
                self.num_clips,
                self.frames_per_clip,
                self.frame_step,
            )
            if self.clip_video
            else []
        )

        video_info.video = VideoCondition(
            video=rgb[None],
            length=torch.tensor([rgb_nframes]),
            clip_indices=[clip_indices],
            sample_rate=[video_meta["video_fps"]],
            path=[video_path],
            seek_time=[info.seek_time],
        )

        video_info.self_wav = WavCondition(
            wav=wav[None],
            length=torch.tensor([info.n_frames]),
            sample_rate=[video_info.sample_rate],
            path=[info.meta.path],
            seek_time=[info.seek_time],
        )

        return wav, video_info


def _get_clip_indices(
    video_len_in_samples: int,
    num_clips: int,
    frames_per_clip: int,
    frame_step: int,
    random_clip_sampling: bool = False,
    allow_clip_overlap: bool = True,
) -> list:
    # Partition video into equal sized segments and sample each clip
    # from a different segment
    partition_len = video_len_in_samples // num_clips
    clip_len = int(frames_per_clip * frame_step)

    clip_indices = []
    for i in range(num_clips):

        if partition_len > clip_len:
            # TODO: If partition_len > clip len, then sample a random window of
            # clip_len frames within the segment
            end_indx = clip_len
            if random_clip_sampling:
                end_indx = np.random.randint(clip_len, partition_len)
            start_indx = end_indx - clip_len
            indices = np.linspace(start_indx, end_indx, num=frames_per_clip)
            indices = np.clip(indices, start_indx, end_indx - 1).astype(np.int64)
            # --
            indices = indices + i * partition_len
        else:
            # TODO: If partition overlap not allowed and partition_len < clip_len
            # then repeatedly append the last frame in the segment until
            # we reach the desired clip length
            if allow_clip_overlap:
                indices = np.linspace(0, partition_len, num=partition_len // frame_step)
                indices = np.concatenate(
                    (
                        indices,
                        np.ones(frames_per_clip - partition_len // frame_step)
                        * partition_len,
                    )
                )
                indices = np.clip(indices, 0, partition_len - 1).astype(np.int64)
                # --
                indices = indices + i * partition_len

            # If partition overlap is allowed and partition_len < clip_len
            # then start_indx of segment i+1 will lie within segment i
            else:
                sample_len = min(clip_len, video_len_in_samples) - 1
                indices = np.linspace(0, sample_len, num=sample_len // frame_step)
                indices = np.concatenate(
                    (
                        indices,
                        np.ones(frames_per_clip - sample_len // frame_step)
                        * sample_len,
                    )
                )
                indices = np.clip(indices, 0, sample_len - 1).astype(np.int64)
                # --
                clip_step = 0
                if video_len_in_samples > clip_len:
                    clip_step = (video_len_in_samples - clip_len) // (num_clips - 1)
                indices = indices + i * clip_step

        clip_indices.append(indices)
    return clip_indices
