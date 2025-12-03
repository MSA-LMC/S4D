# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.distributed as dist

from typing import Callable, Iterable, Optional

from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset

from .omni_dataset import OmniDataset

# me: add for oversamplering
from .sampler import ImbalancedDatasetSampler, DistributedSamplerWrapper


def is_distributed_training_run() -> bool:
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and (torch.distributed.get_world_size() > 1)
    )

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


class TorchDataset(OmniDataset):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
        pin_memory: bool,
        drop_last: bool,
        collate_fn: Optional[Callable] = None,
        worker_init_fn: Optional[Callable] = None,
        oversamplering=False
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn
        num_tasks = get_world_size()
        global_rank = get_rank()
        assert not isinstance(
            self.dataset, IterableDataset), "Not supported yet"
        if is_distributed_training_run():
            if not oversamplering:
                self.sampler = DistributedSampler(
                    self.dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
            else:
                self.sampler = DistributedSamplerWrapper(
                    ImbalancedDatasetSampler(self.dataset), num_replicas=num_tasks, rank=global_rank, shuffle=self.shuffle)
        else:
            self.sampler = None

    def get_loader(self, epoch) -> Iterable:
        self.sampler.set_epoch(epoch) if self.sampler else None

        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            sampler=self.sampler,
            collate_fn=self.collate_fn,
            worker_init_fn=self.worker_init_fn,
        )
