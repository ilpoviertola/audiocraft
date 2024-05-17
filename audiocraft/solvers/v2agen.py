# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import torch

from . import builders, musicgen
from .. import models


class V2AGenSolver(musicgen.MusicGenSolver):
    """Solver for AudioGen re-implementation training task.

    Note that this implementation does not strictly follows
    the method proposed in https://arxiv.org/abs/2209.15352
    but is derived from MusicGen's training pipeline.

    More information can be found in the AudioGen model card.
    """

    DATASET_TYPE: builders.DatasetType = builders.DatasetType.SOUND

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
