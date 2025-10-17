# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from abc import ABC, abstractmethod
from typing import Any, List
import os

import numpy as np
import torch
import torchvision.transforms.functional as tvf
from iopath.common.file_io import g_pathmgr
from omnivision.data.api import VisionSample
from omnivision.utils.data import (
    get_mean_image,
    IdentityTransform,
    SharedMemoryNumpyLoader,
)
from PIL import Image
import random
from pytorchvideo.data.encoded_video import EncodedVideo
from torch.utils.data import Dataset
import decord
from omnivision.utils.distributed import get_rank, is_distributed_training_run

IDENTITY_TRANSFORM = IdentityTransform()
DEFAULT_SPATIAL_SIZE = 224


class PathDataset(Dataset, ABC):
    def __init__(
        self,
        path_file_list: List[str],
        label_file_list: List[str],
        remove_prefix="",
        new_prefix="",
        remove_suffix="",
        new_suffix="",
        transforms=None,
        data_path=None,
    ):
        """Creates a dataset where the metadata is stored in a numpy file.

        path_file_list: A list of paths which contain the path metadata file. Each element
            is tried (in order) until a file that exists is found. That file is then
            used to read the metadata.
        label_file_list: A list of paths which contain the label metadata file. Each element
            is tried (in order) until a file that exists is found. That file is then
            used to read the metadata.
        """
        self.is_initialized = False
        self.path_file_list = path_file_list
        self.label_file_list = label_file_list
        self.transforms = [] if transforms is None else transforms

        self.remove_prefix = remove_prefix
        self.new_prefix = new_prefix
        self.remove_suffix = remove_suffix
        self.new_suffix = new_suffix

        self.paths = None
        self.labels = None
        self.file_idx = None
        self.data_path = data_path
        self.device = f'cuda:{get_rank()}' if is_distributed_training_run() else 'cpu'
        # used for shared memory
        self.label_sm_loader = SharedMemoryNumpyLoader()
        self.path_sm_loader = SharedMemoryNumpyLoader()

        self._load_data()
        self.num_samples = len(self.paths)
        assert len(self.paths) == len(
            self.labels
        ), f"Paths ({len(self.paths)}) != labels ({len(self.labels)})"
        logging.info(
            f"Created dataset from {self.path_file_list} of length: {self.num_samples}"
        )

    def _load_data(self):
        logging.info(f"Loading {self.label_file_list} with shared memory")
        self.labels, label_file_idx = self.label_sm_loader.load(self.label_file_list)
        logging.info(f"Loading {self.path_file_list} with shared memory")
        self.paths, path_file_idx = self.path_sm_loader.load(self.path_file_list)
        assert (
            label_file_idx == path_file_idx
        ), "Label file and path file were not found at the same index"
        self.is_initialized = True
        self.file_idx = path_file_idx

    def _replace_path_prefix(self, path, replace_prefix, new_prefix):
        if replace_prefix == "":
            path = new_prefix + path
        elif path.startswith(replace_prefix):
            return new_prefix + path[len(replace_prefix) :]
        else:
            raise ValueError(f"Cannot replace `{replace_prefix}`` prefix in `{path}`")
        return path

    def _replace_path_suffix(self, path, replace_suffix, new_suffix):
        if replace_suffix == "":
            path = path + new_suffix
        elif path.endswith(replace_suffix):
            return path[: -len(replace_suffix)] + new_suffix
        else:
            raise ValueError(f"Cannot replace `{replace_suffix}`` suffix in `{path}`")
        return path

    def __len__(self):
        return self.num_samples

    @abstractmethod
    def default_generator(self):
        pass

    @abstractmethod
    def load_object(self, path):
        pass

    def _get_path(self, idx):
        path = self._replace_path_prefix(
            self.paths[idx],
            replace_prefix=self.remove_prefix,
            new_prefix=self.new_prefix,
        )
        path = self._replace_path_suffix(
            path, replace_suffix=self.remove_suffix, new_suffix=self.new_suffix
        )
        if self.data_path is not None:
            path = os.path.join(self.data_path, path)
        return path

    def try_load_object(self, idx):
        is_success = True
        try:
            data = self.load_object(self._get_path(idx))
        except Exception:
            logging.warning(
                f"Couldn't load: {self.paths[idx]}. Exception:", exc_info=True
            )
            is_success = False
            data = self.default_generator()
        return data, is_success

    def get_label(self, idx):
        return None if self.labels is None else self.labels[idx]
    def get_labels(self):
        return self.labels
    @staticmethod
    def create_sample(idx, data, label, is_success):
        return VisionSample(
            vision=data, label=int(label), data_idx=idx, data_valid=is_success
        )

    def apply_transforms(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def __getitem__(self, idx):
        data, is_success = self.try_load_object(idx)
        label = self.get_label(idx)
        sample = self.create_sample(idx, data, label, is_success)
        sample = self.apply_transforms(sample)
        return sample


class ImagePathDataset(PathDataset):
    def __init__(
        self,
        dataset_type=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # me: new added for VoxCeleb2
        self.is_voxceleb2 = False
        self.crop_idxs = None
        if dataset_type and'voxceleb2' in dataset_type.lower():
            self.is_voxceleb2 = True
            image_size = 160
            if image_size == 192:
                self.crop_idxs = ((0, 192), (16, 208))
                logging.info(f"==> Note: use crop_idxs={self.crop_idxs} for VoxCeleb2!!!")
            elif image_size <= 160: # me: old is == 160
                self.crop_idxs = ((0, 160), (32, 192))
                logging.info(f"==> Note: use crop_idxs={self.crop_idxs} for VoxCeleb2!!!")


    def default_generator(self):
        return get_mean_image(DEFAULT_SPATIAL_SIZE)

    def load_object(self, path) -> Image:
        with g_pathmgr.open(path, "rb") as fopen:
            frame = Image.open(fopen).convert("RGB")
            if self.is_voxceleb2 and self.crop_idxs:
                frame = frame.crop((self.crop_idxs[1][0], self.crop_idxs[0][0], self.crop_idxs[1][1], self.crop_idxs[0][1]))
            return frame
            

class ImageWithDepthPathDataset(ImagePathDataset):
    def __init__(
        self,
        depth_path_file_list: List[str],
        *args,
        remove_depth_prefix="",
        new_depth_prefix="",
        remove_depth_suffix="",
        new_depth_suffix="",
        **kwargs,
    ):
        """
        Shared Memory dataloader for RGB+Depth datasets.
        """
        super().__init__(*args, **kwargs)

        self.depth_path_file_list = depth_path_file_list

        self.remove_depth_prefix = remove_depth_prefix
        self.new_depth_prefix = new_depth_prefix
        self.remove_depth_suffix = remove_depth_suffix
        self.new_depth_suffix = new_depth_suffix

        self.depth_path_sm_loader = SharedMemoryNumpyLoader()

        logging.info(f"Loading {self.depth_path_file_list} with shared memory")
        self.depth_paths, depth_file_idx = self.depth_path_sm_loader.load(
            self.depth_path_file_list
        )

        assert (
            depth_file_idx == self.file_idx
        ), "Depth file and path file were not found at the same index"

    def _load_depth(self, image_path):
        """
        Returns:
            A (H, W, 1) tensor
        """
        with g_pathmgr.open(image_path, "rb") as fopen:
            # Depth is being saved as a .pt file instead
            # of as an image
            return torch.load(fopen)

    def _get_depth_path(self, idx):
        path = self._replace_path_prefix(
            self.depth_paths[idx],
            replace_prefix=self.remove_depth_prefix,
            new_prefix=self.new_depth_prefix,
        )
        path = self._replace_path_suffix(
            path,
            replace_suffix=self.remove_depth_suffix,
            new_suffix=self.new_depth_suffix,
        )
        return path

    def default_generator(self):
        image = get_mean_image(DEFAULT_SPATIAL_SIZE)
        depth = torch.zeros(
            (1, DEFAULT_SPATIAL_SIZE, DEFAULT_SPATIAL_SIZE), dtype=torch.float32
        )
        return torch.cat([tvf.to_tensor(image), depth], dim=0)

    def try_load_object(self, idx):
        image, is_success = super().try_load_object(idx)
        if is_success:
            try:
                depth = self._load_depth(self._get_depth_path(idx))
                if depth.ndim == 2:
                    depth = depth[None, ...]  # (1, H, W)
                image_with_depth = torch.cat(
                    [tvf.to_tensor(image), depth], dim=0
                )  # (4, H, W)
            except Exception:
                logging.warning(
                    f"Couldn't load depth image: {self.depth_paths[idx]}. Exception:",
                    exc_info=True,
                )
                is_success = False

        if not is_success:
            image_with_depth = self.default_generator()

        return image_with_depth, is_success

# TODO: 修改成 VideoMAE 的读取形式，使用 Decord读取指定的帧
class VideoPathDataset(PathDataset):
    def __init__(
        self,
        clip_sampler,
        frame_sampler,
        decoder,
        normalize_to_0_1,
        all_segments=True,
        dataset_type=None,
        *args,
        decoder_kwargs=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.clip_sampler = clip_sampler
        self.frame_sampler = frame_sampler
        self.decoder = decoder
        self.normalize_to_0_1 = normalize_to_0_1
        self.decoder_kwargs = {} if decoder_kwargs is None else decoder_kwargs

        self.all_segments = all_segments
        # me: new added for VoxCeleb2
        self.is_voxceleb2 = False
        self.crop_idxs = None
        if dataset_type and'voxceleb2' in dataset_type.lower():
            self.is_voxceleb2 = True
            image_size = 160
            if image_size == 192:
                self.crop_idxs = ((0, 192), (16, 208))
                logging.info(f"==> Note: use crop_idxs={self.crop_idxs} for VoxCeleb2!!!")
            elif image_size <= 160: # me: old is == 160
                self.crop_idxs = ((0, 160), (32, 192))
                logging.info(f"==> Note: use crop_idxs={self.crop_idxs} for VoxCeleb2!!!")

    def _get_video_object(self, path):
        return EncodedVideo.from_path(
            path, decoder=self.decoder, decode_audio=False, **self.decoder_kwargs
        )

    def load_object(self, path) -> List[torch.Tensor]:
        """
        Returns:
            A (C, T, H, W) tensor.
        """
        video = self._get_video_object(path)
        # Read out all clips in this video
        all_clips_timepoints = []
        is_last_clip = False
        end = 0.0
        while not is_last_clip:
            start, end, _, _, is_last_clip = self.clip_sampler(
                end, video.duration, annotation=None
            )
            all_clips_timepoints.append((start, end))
        all_frames = []
        for clip_timepoints in all_clips_timepoints:
            # Read the clip, get frames
            clip = video.get_clip(clip_timepoints[0], clip_timepoints[1])["video"] #一次读取的帧数太多了，浪费训练时间
            if clip is None:
                logging.error(
                    "Got a None clip. Make sure the clip timepoints "
                    "are long enough: %s",
                    clip_timepoints,
                )  
            frames = self.frame_sampler(clip)
            if self.normalize_to_0_1:
                frames = frames / 255.0  # since this is float, need 0-1

            if self.is_voxceleb2 and self.crop_idxs:
                frames = frames[:, :, self.crop_idxs[0][0]:self.crop_idxs[0][1], self.crop_idxs[1][0]:self.crop_idxs[1][1]]                   
            all_frames.append(frames)
            if not self.all_segments:
                break
        if len(all_frames) == 1:
            # When only one clip is sampled (eg at training time), remove the
            # outermost list object so it can work with default collators etc.
            all_frames = all_frames[0]
        return all_frames

    def default_generator(self):
        dummy = (
            torch.ones(
                (
                    3,
                    self.frame_sampler._num_samples,
                    DEFAULT_SPATIAL_SIZE,
                    DEFAULT_SPATIAL_SIZE,
                )
            )
            * 0.5
        )
        if hasattr(self.clip_sampler, "_clips_per_video"):
            return [dummy] * self.clip_sampler._clips_per_video
        return dummy

class VideoPathDatasetDecode(PathDataset):
    def __init__(
        self,
        num_segments=1,
        num_crop=1,
        new_length=1,
        new_step=4,
        temporal_jitter=False,
        lazy_init=False,
        decoder='decord',
        normalize_to_0_1=True,
        dataset_type=None,
        *args,
        decoder_kwargs=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.new_length = new_length
        self.new_step = new_step
        self.skip_length = new_length * new_step
        self.temporal_jitter = temporal_jitter
        self.lazy_init = lazy_init

        self.decoder = decoder
        self.normalize_to_0_1 = normalize_to_0_1
        self.decoder_kwargs = {} if decoder_kwargs is None else decoder_kwargs
        # me: new added for VoxCeleb2
        self.is_voxceleb2 = False
        self.crop_idxs = None
        if dataset_type and'voxceleb2' in dataset_type.lower():
            self.is_voxceleb2 = True
            image_size = 160
            if image_size == 192:
                self.crop_idxs = ((0, 192), (16, 208))
                print(f"==> Note: use crop_idxs={self.crop_idxs} for VoxCeleb2!!!")
            elif image_size <= 160: # me: old is == 160
                self.crop_idxs = ((0, 160), (32, 192))
                print(f"==> Note: use crop_idxs={self.crop_idxs} for VoxCeleb2!!!")

    def _get_video_object(self, path):
        try:
            decord_vr = decord.VideoReader(path, num_threads=1)        
        except Exception as e:
            next_idx = random.randint(0, self.__len__() - 1)
            print(f"==> Exception '{e}' occurred when processed '{path}', move to random next one (idx={next_idx}).")
            return self.__getitem__(next_idx)
        return decord_vr

    def load_object(self, path) -> List[torch.Tensor]:
        """
        Returns:
            A (C, T, H, W) tensor.
        """
        decord_vr = self._get_video_object(path)
        duration = len(decord_vr)
        segment_indices, skip_offsets = self._sample_train_indices(duration)

        frames = self._video_TSN_decord_batch_loader(path, decord_vr, duration, segment_indices, skip_offsets)
        # to torch tensor
        frames = torch.cat(
                    [tvf.to_tensor(image).unsqueeze(1) for image in frames], dim=1
                )  
        return frames.to(self.device)

    def _sample_train_indices(self, num_frames):
        average_duration = (num_frames - self.skip_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)),
                                  average_duration)
            offsets = offsets + np.random.randint(average_duration,
                                                  size=self.num_segments)
        elif num_frames > max(self.num_segments, self.skip_length):
            offsets = np.sort(np.random.randint(
                num_frames - self.skip_length + 1,
                size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))

        if self.temporal_jitter:
            skip_offsets = np.random.randint(
                self.new_step, size=self.skip_length // self.new_step)
        else:
            skip_offsets = np.zeros(
                self.skip_length // self.new_step, dtype=int)
        return offsets + 1, skip_offsets


    def _video_TSN_decord_batch_loader(self, directory, video_reader, duration, indices, skip_offsets):
        sampled_list = []
        frame_id_list = []
        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
                if offset + skip_offsets[i] <= duration:
                    frame_id = offset + skip_offsets[i] - 1
                else:
                    frame_id = offset - 1
                frame_id_list.append(frame_id)
                if offset + self.new_step < duration:
                    offset += self.new_step
        try:
            video_data = video_reader.get_batch(frame_id_list).asnumpy()
            if self.is_voxceleb2 and self.crop_idxs is not None:
                sampled_list = [Image.fromarray(video_data[vid, self.crop_idxs[0][0]:self.crop_idxs[0][1], self.crop_idxs[1][0]:self.crop_idxs[1][1], :]).convert('RGB') for vid, _ in enumerate(frame_id_list)]
            else:
                sampled_list = [Image.fromarray(video_data[vid, :, :, :]).convert('RGB') for vid, _ in enumerate(frame_id_list)]
        except:
            raise RuntimeError('Error occured in reading frames {} from video {} of duration {}.'.format(frame_id_list, directory, duration))
        return sampled_list

    def default_generator(self):
        dummy = (
            torch.ones(
                (
                    3,
                    self.frame_sampler._num_samples,
                    DEFAULT_SPATIAL_SIZE,
                    DEFAULT_SPATIAL_SIZE,
                )
            )
            * 0.5
        )
        if hasattr(self.clip_sampler, "_clips_per_video"):
            return [dummy] * self.clip_sampler._clips_per_video
        return dummy


    def __getitem__(self, idx):
        data, is_success = self.try_load_object(idx)
        label = self.get_label(idx)
        sample1 = self.create_sample(idx, data, label, is_success)
        sample2 = self.create_sample(idx, data, label, is_success)
        sample1 = self.apply_transforms(sample1)
        sample2 = self.apply_transforms(sample2)
        
        if not isinstance(sample1, list):
            sample1.vision = torch.cat([sample1.vision, sample2.vision], dim=0)
            sample1.mask = torch.cat([sample1.mask, sample2.mask], dim=0)
            return sample1
        
        sample=[]
        for e1, e2 in zip(sample1, sample2):
            e1.vision = torch.cat([e1.vision, e2.vision], dim=1)
            e1.mask = torch.cat([e1.mask, e2.mask], dim=1)
            sample.append(e1)
            
        return sample