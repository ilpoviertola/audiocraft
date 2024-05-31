import typing as tp
import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf

from .encodec import CompressionModel
from .genmodel import BaseGenModel
from .lm import LMModel
from .loaders import load_compression_model, load_lm_model
from .. import models
from ..utils import checkpoint
from ..modules.conditioners import ConditioningAttributes, VideoCondition
from ..utils.transforms import get_jepa_transforms_validation
from ..data.video_dataset import _get_clip_indices


VideoList = tp.List[tp.Optional[torch.Tensor]]
VideoType = tp.Union[torch.Tensor, VideoList]
logger = logging.getLogger(__name__)


class V2AGen(BaseGenModel):
    def __init__(
        self,
        name: str,
        compression_model: CompressionModel,
        lm: LMModel,
        max_duration: tp.Optional[float] = None,
    ):
        super().__init__(name, compression_model, lm, max_duration)
        self.set_generation_params(
            duration=9.60
        )  # default duration 15 * 0.64 seconds (VJEPA clip length)
        self.video_transforms = get_jepa_transforms_validation()

    @property
    def vfps(self):
        return self.lm.condition_provider.conditioners["video"].sample_rate

    @property
    def patch_size(self):
        return self.lm.condition_provider.conditioners[
            "video"
        ].jepa.model.model.patch_size

    @property
    def frame_step(self):
        return self.lm.cfg.dataset.frame_step

    @staticmethod
    def get_pretrained(filepath: str, device=None):
        if device is None:
            if torch.cuda.device_count():
                device = "cuda"
            else:
                device = "cpu"

        compression_model = models.CompressionModel.get_pretrained(
            "facebook/encodec_24kHz", device=device
        )
        # lm = load_lm_model(filepath, device=device)
        pkg = torch.load(filepath, map_location=device)
        cfg = OmegaConf.create(pkg["xp.cfg"])
        lm = models.builders.get_lm_model(cfg).to(device)
        lm.load_state_dict(pkg["best_state"]["model"])
        lm.cfg = cfg
        compression_model.set_num_codebooks(lm.n_q)
        assert (
            "video" in lm.condition_provider.conditioners
        ), "Language model does not support video conditioning."
        return V2AGen("v2agen", compression_model, lm)

    def set_generation_params(
        self,
        use_sampling: bool = True,
        top_k: int = 125,
        top_p: float = 0.0,
        temperature: float = 1.0,
        duration: float = 10.0,
        cfg_coef: float = 3.0,
        two_step_cfg: bool = False,
        extend_stride: float = 0.48,
    ):
        """Set the generation parameters for AudioGen.

        Args:
            use_sampling (bool, optional): Use sampling if True, else do argmax decoding. Defaults to True.
            top_k (int, optional): top_k used for sampling. Defaults to 250.
            top_p (float, optional): top_p used for sampling, when set to 0 top_k is used. Defaults to 0.0.
            temperature (float, optional): Softmax temperature parameter. Defaults to 1.0.
            duration (float, optional): Duration of the generated waveform. Defaults to 10.0.
            cfg_coef (float, optional): Coefficient used for classifier free guidance. Defaults to 3.0.
            two_step_cfg (bool, optional): If True, performs 2 forward for Classifier Free Guidance,
                instead of batching together the two. This has some impact on how things
                are padded but seems to have little impact in practice.
            extend_stride: when doing extended generation (i.e. more than 10 seconds), by how much
                should we extend the audio each time. Larger values will mean less context is
                preserved, and shorter value will require extra computations.
        """
        assert (
            extend_stride < self.max_duration
        ), "Cannot stride by more than max generation duration."
        self.extend_stride = extend_stride
        self.duration = duration
        self.generation_params = {
            "use_sampling": use_sampling,
            "temp": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "cfg_coef": cfg_coef,
            "two_step_cfg": two_step_cfg,
        }

    def generate_with_video(
        self,
        videos: VideoType,
        progress: bool = False,
        return_tokens: bool = False,
    ):
        """Generate audio from video.

        Args:
            videos (VideoType): A batch of videos used for generation. Should have shape [B, T, C, H, W] or [T, C, H, W].
                Can also be a list of [T, C, H, W] tensors.
            video_fps (int): Video sample rate.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
            return_tokens (bool, optional): Whether to return the tokens with audio. Defaults to False.

        Raises:
            ValueError: If videos shape is not [B, T, C, H, W] or [T, C, H, W].
        """
        if isinstance(videos, torch.Tensor):
            if videos.dim() == 4:
                videos = videos[None]
            if videos.dim() != 5:
                raise ValueError(
                    "Videos should have shape [B, T, C, H, W] or [T, C, H, W]"
                )
        else:
            for video in videos:
                if video is not None:
                    assert (
                        video.dim() == 4
                    ), "Videos should have shape [T, C, H, W] if provided as a list"

        frames = int(self.duration * self.vfps)
        videos = [
            self.video_transforms(video)[:, :frames] if video is not None else None
            for video in videos
        ]
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(
            descriptions=[], prompt=None, videos=videos
        )
        assert prompt_tokens is None
        tokens = self._generate_tokens(attributes, prompt_tokens, progress=progress)
        if return_tokens:
            return self.generate_audio(tokens), tokens
        return self.generate_audio(tokens)

    @torch.no_grad()
    def _prepare_tokens_and_attributes(
        self,
        descriptions: tp.Sequence[tp.Optional[str]],
        prompt: tp.Optional[torch.Tensor],
        videos: tp.Optional[VideoType] = None,
    ) -> tp.Tuple[tp.List[ConditioningAttributes], tp.Optional[torch.Tensor]]:
        if descriptions:
            logger.warning("Descriptions are not supported for now.")

        assert videos is not None, "Videos are required for generation."
        if "video" not in self.lm.condition_provider.conditioners:
            raise RuntimeError("Language model does not support video conditioning.")
        attributes = []
        for video in videos:
            attr = ConditioningAttributes()
            if video is None:
                attr.video["video"] = VideoCondition(
                    torch.zeros((1, 3, 1, 1, 1), device=self.device),
                    torch.tensor([0], device=self.device),
                    [self.vfps],
                    [None],
                    [0.0],
                    [],
                )
            else:
                attr.video["video"] = VideoCondition(
                    video[None].to(device=self.device),
                    torch.tensor([video.shape[1]], device=self.device),
                    [self.vfps],
                    [None],
                    [0.0],
                    [
                        _get_clip_indices(
                            video.shape[1],
                            video.shape[1] // self.patch_size,
                            self.patch_size,
                            self.frame_step,
                        )
                    ],
                )
            attributes.append(attr)

        if prompt is not None:
            if videos is not None:
                assert len(videos) == len(
                    prompt
                ), "Prompt and nb. descriptions doesn't match"
            prompt = prompt.to(self.device)
            prompt_tokens, scale = self.compression_model.encode(prompt)
            assert scale is None
        else:
            prompt_tokens = None
        return attributes, prompt_tokens

    def _generate_tokens(
        self,
        attributes: tp.List[ConditioningAttributes],
        prompt_tokens: tp.Optional[torch.Tensor],
        progress: bool = False,
    ) -> torch.Tensor:
        """Generate discrete audio tokens given audio prompt and/or conditions.

        Args:
            attributes (list of ConditioningAttributes): Conditions used for generation (text/melody).
            prompt_tokens (torch.Tensor, optional): Audio prompt used for continuation.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        Returns:
            torch.Tensor: Generated audio, of shape [B, T, C], T is defined by the generation params.
        """
        total_gen_len = int(self.duration * self.frame_rate)
        max_prompt_len = int(min(self.duration, self.max_duration) * self.frame_rate)
        current_gen_offset: int = 0

        def _progress_callback(generated_tokens: int, tokens_to_generate: int):
            generated_tokens += current_gen_offset
            if self._progress_callback is not None:
                # Note that total_gen_len might be quite wrong depending on the
                # codebook pattern used, but with delay it is almost accurate.
                self._progress_callback(generated_tokens, tokens_to_generate)
            else:
                print(f"{generated_tokens: 6d} / {tokens_to_generate: 6d}", end="\r")

        if prompt_tokens is not None:
            assert (
                max_prompt_len >= prompt_tokens.shape[-1]
            ), "Prompt is longer than audio to generate"

        callback = None
        if progress:
            callback = _progress_callback

        if self.duration <= self.max_duration:
            # generate by sampling from LM, simple case.
            with self.autocast:
                gen_tokens = self.lm.generate(
                    prompt_tokens,
                    attributes,
                    callback=callback,
                    max_gen_len=total_gen_len,
                    **self.generation_params,
                )

        else:
            # now this gets a bit messier, we need to handle prompts,
            # melody conditioning etc.
            ref_videos = [attr.video["video"] for attr in attributes]
            all_tokens = []
            if prompt_tokens is None:
                prompt_length = 0
            else:
                all_tokens.append(prompt_tokens)
                prompt_length = prompt_tokens.shape[-1]

            assert (
                self.extend_stride is not None
            ), "Stride should be defined to generate beyond max_duration"
            assert (
                self.extend_stride < self.max_duration
            ), "Cannot stride by more than max generation duration."
            stride_tokens = int(self.frame_rate * self.extend_stride)

            while current_gen_offset + prompt_length < total_gen_len:
                time_offset = current_gen_offset / self.frame_rate
                chunk_duration = min(self.duration - time_offset, self.max_duration)
                max_gen_len = int(chunk_duration * self.frame_rate)
                for attr, ref_video in zip(attributes, ref_videos):
                    video_length = ref_video.length.item()
                    if video_length == 0:
                        continue
                    # We will extend the wav periodically if it not long enough.
                    # we have to do it here rather than in conditioners.py as otherwise
                    # we wouldn't have the full wav.
                    initial_position = int(time_offset * self.vfps)
                    video_target_length = int(self.max_duration * self.vfps)
                    positions = torch.arange(
                        initial_position,
                        initial_position + video_target_length,
                        device=self.device,
                    )
                    attr.video["video"] = VideoCondition(
                        ref_video[0][:, :, positions % video_length, ...],
                        torch.full_like(ref_video[1], video_target_length),
                        [self.sample_rate] * ref_video[0].size(0),
                        [None],
                        [time_offset],
                        [
                            _get_clip_indices(
                                video_target_length,
                                video_target_length // self.patch_size,
                                self.patch_size,
                                self.frame_step,
                            )
                        ],
                    )
                with self.autocast:
                    gen_tokens = self.lm.generate(
                        prompt_tokens,
                        attributes,
                        callback=callback,
                        max_gen_len=max_gen_len,
                        **self.generation_params,
                    )
                if prompt_tokens is None:
                    all_tokens.append(gen_tokens)
                else:
                    all_tokens.append(gen_tokens[:, :, prompt_tokens.shape[-1] :])
                prompt_tokens = gen_tokens[:, :, stride_tokens:]
                prompt_length = prompt_tokens.shape[-1]
                current_gen_offset += stride_tokens

            gen_tokens = torch.cat(all_tokens, dim=-1)
        return gen_tokens
