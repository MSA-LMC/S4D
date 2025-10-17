# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from omnivision.trainer.omnivision_trainer import OmnivisionTrainer
import contextlib
import json
import logging
import math
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr
from omnivision.data.api import Sample
from omnivision.data.concat_dataset import ConcatDataset
from omnivision.data.torch_dataset import TorchDataset
from omnivision.losses import wrap_base_loss
from omnivision.optim import construct_optimizer
from omnivision.utils.train import (
    AverageMeter,
    copy_data_to_device,
    get_amp_type,
    ProgressMeter,
)


def chunk_batch_for_accum_steps(batch, accum_steps):
    return [get_chunk_from_data(batch, i, accum_steps) for i in range(accum_steps)]


def get_chunk_from_data(data, chunk_id, num_chunks):
    """
    Recursively splits all the tensors inside the passed data object into num_chunks.
    """
    if isinstance(data, torch.Tensor):
        assert len(data) % num_chunks == 0
        start = (len(data) // num_chunks) * chunk_id
        end = (len(data) // num_chunks) * (chunk_id + 1)
        return data[start:end]
    elif isinstance(data, Mapping):
        return {
            key: get_chunk_from_data(value, chunk_id, num_chunks)
            for key, value in data.items()
        }
    elif isinstance(data, Sequence):
        return [get_chunk_from_data(value, chunk_id, num_chunks) for value in data]
    elif isinstance(data, Sample):
        data_cls = type(data)
        data = data.__dict__
        return data_cls(**get_chunk_from_data(data, chunk_id, num_chunks))
    else:
        return data



class VideoMAETrainer(OmnivisionTrainer):

    def _step(self, batch: Any, key: str, phase_type: str):
        x, mask = batch.vision, batch.mask
        # y_hat = self.model({key: batch}, **batch.model_fwd_kwargs)
        y_hat = self.model(x, mask=mask, **batch.model_fwd_kwargs)
        
        # # assert isinstance(y_hat, Mapping)
        # # assert len(y_hat) == 1
        # key, y_hat = y_hat.popitem()
        batch_size = batch.label.shape[0]
        loss_str = f"Losses/{phase_type}_{key}_loss"
        if phase_type == "train":
            loss, y_hat = self.loss[key](y_hat, batch)
            self.logger.log(
                os.path.join("Step", loss_str),
                loss,
                self.steps[phase_type],
            )

        metrics_result = self._compute_metrics(y_hat, batch.label, phase_type, key)
        self.logger.log_dict(
            {os.path.join("Step", k): v for k, v in metrics_result.items()},
            self.steps[phase_type],
        )

        self.steps[phase_type] += 1

        return {loss_str: loss}, metrics_result, batch_size
