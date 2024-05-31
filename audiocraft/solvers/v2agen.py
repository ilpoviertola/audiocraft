# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import torch
import flashy

from . import builders, musicgen
from .. import models
from ..data.audio_dataset import AudioDataset
from ..modules.conditioners import (
    JointEmbedCondition,
    SegmentWithAttributes,
    WavCondition,
    VideoCondition,
)

from ..utils.samples.manager import SampleManager
from ..utils.utils import get_dataset_from_loader, is_jsonable


class V2AGenSolver(musicgen.MusicGenSolver):
    """Solver for AudioGen re-implementation training task.

    Note that this implementation does not strictly follows
    the method proposed in https://arxiv.org/abs/2209.15352
    but is derived from MusicGen's training pipeline.

    More information can be found in the AudioGen model card.
    """

    DATASET_TYPE: builders.DatasetType = builders.DatasetType.VIDEO

    def build_model(self) -> None:
        """Instantiate models and optimizer."""
        # we can potentially not use all quantizers with which the EnCodec model was trained
        # (e.g. we trained the model with quantizers dropout)
        self.compression_model = models.CompressionModel.get_pretrained(
            self.cfg.compression_model_checkpoint, device=self.device
        )
        self.compression_model.set_num_codebooks(self.cfg.transformer_lm.n_q)
        assert self.compression_model.sample_rate == self.cfg.sample_rate, (
            f"Compression model sample rate is {self.compression_model.sample_rate} but "
            f"Solver sample rate is {self.cfg.sample_rate}."
        )
        # ensure we have matching configuration between LM and compression model
        assert self.cfg.transformer_lm.card == self.compression_model.cardinality, (
            "Cardinalities of the LM and compression model don't match: ",
            f"LM cardinality is {self.cfg.transformer_lm.card} vs ",
            f"compression model cardinality is {self.compression_model.cardinality}",
        )
        assert self.cfg.transformer_lm.n_q == self.compression_model.num_codebooks, (
            "Numbers of codebooks of the LM and compression models don't match: ",
            f"LM number of codebooks is {self.cfg.transformer_lm.n_q} vs ",
            f"compression model numer of codebooks is {self.compression_model.num_codebooks}",
        )
        self.logger.info(
            "Compression model has %d codebooks with %d cardinality, and a framerate of %d",
            self.compression_model.num_codebooks,
            self.compression_model.cardinality,
            self.compression_model.frame_rate,
        )
        # instantiate LM model
        self.model: models.LMModel = models.builders.get_lm_model(self.cfg).to(
            self.device
        )
        if self.cfg.fsdp.use:
            assert not self.cfg.autocast, "Cannot use autocast with fsdp"
            self.model = self.wrap_with_fsdp(self.model)
        self.register_ema("model")
        # initialize optimization
        self.optimizer = builders.get_optimizer(
            builders.get_optim_parameter_groups(self.model), self.cfg.optim
        )
        self.lr_scheduler = builders.get_lr_scheduler(
            self.optimizer, self.cfg.schedule, self.total_updates
        )
        self.register_stateful("model", "optimizer", "lr_scheduler")
        self.register_best_state("model")
        self.autocast_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[
            self.cfg.autocast_dtype
        ]
        self.scaler: tp.Optional[torch.cuda.amp.GradScaler] = None
        if self.cfg.fsdp.use:
            need_scaler = self.cfg.fsdp.param_dtype == "float16"
        else:
            need_scaler = self.cfg.autocast and self.autocast_dtype is torch.float16
        if need_scaler:
            if self.cfg.fsdp.use:
                from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

                self.scaler = ShardedGradScaler()  # type: ignore
            else:
                self.scaler = torch.cuda.amp.GradScaler()
            self.register_stateful("scaler")

    def generate_audio(self) -> dict:
        """Audio generation stage."""
        generate_stage_name = f"{self.current_stage}"
        sample_manager = SampleManager(self.xp)
        self.logger.info(f"Generating samples in {sample_manager.base_folder}")
        loader = self.dataloaders["generate"]
        updates = len(loader)
        lp = self.log_progress(
            generate_stage_name, loader, total=updates, updates=self.log_updates
        )

        dataset = get_dataset_from_loader(loader)
        dataset_duration = dataset.segment_duration
        assert dataset_duration is not None
        assert isinstance(dataset, AudioDataset)
        target_duration = self.cfg.generate.lm.gen_duration
        prompt_duration = self.cfg.generate.lm.prompt_duration
        if target_duration is None:
            target_duration = dataset_duration
        if prompt_duration is None:
            prompt_duration = dataset_duration / 4
        assert prompt_duration < dataset_duration, (
            f"Specified prompt duration ({prompt_duration}s) is longer",
            f" than reference audio duration ({dataset_duration}s)",
        )

        def get_hydrated_conditions(meta: tp.List[SegmentWithAttributes]):
            hydrated_conditions = []
            for sample in [x.to_condition_attributes() for x in meta]:
                cond_dict = {}
                for cond_type in sample.__annotations__.keys():
                    for cond_key, cond_val in getattr(sample, cond_type).items():
                        if (
                            cond_key
                            not in self.model.condition_provider.conditioners.keys()
                        ):
                            continue
                        if is_jsonable(cond_val):
                            cond_dict[cond_key] = cond_val
                        elif isinstance(cond_val, WavCondition):
                            cond_dict[cond_key] = cond_val.path
                        elif isinstance(cond_val, JointEmbedCondition):
                            cond_dict[cond_key] = (
                                cond_val.text
                            )  # only support text at inference for now
                        elif isinstance(cond_val, VideoCondition):
                            cond_dict[cond_key] = {
                                "path": cond_val.path,
                                "seek_time": cond_val.seek_time,
                            }
                        else:
                            # if we reached this point, it is not clear how to log the condition
                            # so we just log the type.
                            cond_dict[cond_key] = str(type(cond_val))
                            continue
                hydrated_conditions.append(cond_dict)
            return hydrated_conditions

        metrics: dict = {}
        average = flashy.averager()
        for batch in lp:
            audio, meta = batch
            # metadata for sample manager
            hydrated_conditions = get_hydrated_conditions(meta)
            sample_generation_params = {
                **{
                    f"classifier_free_guidance_{k}": v
                    for k, v in self.cfg.classifier_free_guidance.items()
                },
                **self.generation_params,
            }
            if self.cfg.generate.lm.unprompted_samples:
                if self.cfg.generate.lm.gen_gt_samples:
                    # get the ground truth instead of generation
                    self.logger.warn(
                        "Use ground truth instead of audio generation as generate.lm.gen_gt_samples=true"
                    )
                    gen_unprompted_audio = audio
                    rtf = 1.0
                else:
                    gen_unprompted_outputs = self.run_generate_step(
                        batch,
                        gen_duration=target_duration,
                        prompt_duration=None,
                        **self.generation_params,
                    )
                    gen_unprompted_audio = gen_unprompted_outputs["gen_audio"].cpu()
                    rtf = gen_unprompted_outputs["rtf"]
                sample_manager.add_samples(
                    gen_unprompted_audio,
                    self.epoch,
                    hydrated_conditions,
                    ground_truth_wavs=audio,
                    generation_args=sample_generation_params,
                )

            if self.cfg.generate.lm.prompted_samples:
                gen_outputs = self.run_generate_step(
                    batch,
                    gen_duration=target_duration,
                    prompt_duration=prompt_duration,
                    **self.generation_params,
                )
                gen_audio = gen_outputs["gen_audio"].cpu()
                prompt_audio = gen_outputs["prompt_audio"].cpu()
                sample_manager.add_samples(
                    gen_audio,
                    self.epoch,
                    hydrated_conditions,
                    prompt_wavs=prompt_audio,
                    ground_truth_wavs=audio,
                    generation_args=sample_generation_params,
                )

            # if "rtf" in metrics:
            metrics["rtf"] = rtf
            metrics = average(metrics)

        # purely debug business
        # if self.cfg.generate.lm.long_samples.generate:
        #     self.logger.info("Generating long samples")
        #     duration = self.cfg.generate.lm.long_samples.duration
        #     if duration <= self.cfg.dataset.segment_duration:
        #         # no need to generate long samples
        #         flashy.distrib.barrier()
        #         return metrics

        #     cfg = self.cfg.copy()
        #     cfg.datasource = cfg.generate.lm.long_samples.datasource
        #     cfg.dataset.segment_duration = duration
        #     long_video_loader = builders.get_audio_datasets(cfg, self.DATASET_TYPE)
        #     updates = len(long_video_loader)
        #     lp = self.log_progress(
        #         generate_stage_name,
        #         long_video_loader,
        #         total=updates,
        #         updates=self.log_updates,
        #     )
        #     stride = cfg.generate.lm.long_samples.stride
        #     stride = duration / 4 * 3 if stride is None else stride
        #     stride_tokens = int(self.compression_model.frame_rate * stride)
        #     total_gen_len = int(duration * self.compression_model.frame_rate)
        #     max_prompt_len = int(
        #         self.cfg.dataset.segment_duration * self.compression_model.frame_rate
        #     )
        #     vfps = self.model.condition_provider.conditioners["video"].sample_rate
        #     for batch in lp:
        #         audio, meta = batch
        #         # metadata for sample manager
        #         hydrated_conditions = get_hydrated_conditions(meta)
        #         sample_generation_params = {
        #             **{
        #                 f"classifier_free_guidance_{k}": v
        #                 for k, v in self.cfg.classifier_free_guidance.items()
        #             },
        #             **self.generation_params,
        #         }
        #         current_gen_offset = 0
        #         all_tokens = []
        #         prompt_length = 0
        #         while current_gen_offset + prompt_length < total_gen_len:
        #             time_offset = current_gen_offset / self.compression_model.frame_rate
        #             chunk_duration = min(
        #                 duration - time_offset, self.cfg.dataset.segment_duration
        #             )
        #             max_gen_len = int(
        #                 chunk_duration * self.compression_model.frame_rate
        #             )
        #             initial_position = int(time_offset * vfps)
        #             video_target_length = int(self.cfg.dataset.segment_duration * vfps)
        #             positions = torch.arange(
        #                 initial_position,
        #                 initial_position + video_target_length,
        #                 device=self.device,
        #             )
        #             # attributes = []
        #             for attr in meta:
        #                 video_length = attr.video.length.item()
        #                 attr.video = VideoCondition(
        #                     video=attr.video.video[:, :, positions % video_length, ...],
        #                     length=torch.full_like(
        #                         attr.video.length, video_target_length
        #                     ),
        #                     sample_rate=[vfps] * attr.video.video.size(0),
        #                     path=attr.video.path,
        #                     seek_time=[time_offset],
        #                     clip_indices=[
        #                         _get_clip_indices(
        #                             video_target_length,
        #                             video_target_length // 16,
        #                             16,
        #                             1,
        #                         )
        #                     ],
        #                 )
        #             self.generation_params["remove_prompt"] = True
        #             self.generation_params["max_gen_len"] = max_gen_len
        #             gen_outputs = self.run_generate_step(
        #                 batch,
        #                 gen_duration=target_duration,
        #                 prompt_duration=(
        #                     self.cfg.dataset.segment_duration - stride
        #                     if prompt_length != 0
        #                     else None
        #                 ),
        #                 **self.generation_params,
        #             )
        #             gen_tokens = gen_outputs["gen_tokens"]
        #             prompt_tokens = gen_outputs["prompt_tokens"]
        #             if prompt_tokens is None:
        #                 all_tokens.append(gen_tokens)
        #             else:
        #                 all_tokens.append(gen_tokens[:, :, prompt_tokens.shape[-1] :])
        #             prompt_tokens = gen_tokens[:, :, stride_tokens:]
        #             prompt_length = prompt_tokens.shape[-1]
        #             current_gen_offset += stride_tokens

        flashy.distrib.barrier()
        return metrics
