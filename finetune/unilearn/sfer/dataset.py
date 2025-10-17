# -*- coding: utf-8 -*-
import os
import torch
import pandas as pd
import random
from PIL import Image, ImageFile
import logging
import math
import cv2
from operator import itemgetter
from typing import Iterator, List, Optional
from torch.utils.data import Dataset, Sampler, DistributedSampler
from torchvision import transforms
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IAMGES = True
LOG = logging.getLogger(__name__)


class RafDataset(torch.utils.data.Dataset):
    """
    # ################### RAF-DB EmoLabel (7 categories) ###################
        # 1: Surprise
        # 2: Fear
        # 3: Disgust
        # 4: Happiness
        # 5: Sadness
        # 6: Anger
        # 7: Neutral
    """

    def __init__(self, args, annotaion_path, transform=None):
        self.raf_path = args.raf_path
        self.transform = transform

        name_c = 0
        label_c = 1
        dataset = pd.read_csv(annotaion_path, sep=' ', header=None)
        # dataset = df[df[name_c].str.startswith('train')]
        print(dataset.groupby([1]).size())

        # notice the RAF-DB label starts from 1 while label of other dataset starts from 0:
        self.label = dataset.iloc[:, label_c].values - 1
        images_names = dataset.iloc[:, name_c].values
        self.image_paths = []

        for f in images_names:
            f = f.split(".")[0]
            f += '_aligned.jpg'
            file_name = os.path.join(self.raf_path, 'Image/aligned', f)
            self.image_paths.append(file_name)

    def __len__(self):
        return len(self.image_paths)
    def get_labels(self):
        return self.label
    def __getitem__(self, idx):
        label = self.label[idx]
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image)
        return image, label, idx


class SFEW2Dataset(torch.utils.data.Dataset):
    """
    # ################### SFEW EmoLabel (7 categories) ###################
        # 0: Neutral
        # 1: Surprise  
        # 2: Disgust
        # 3: Happy
        # 4: Sad
        # 5: Angry
        # 6: Fear
    """

    def __init__(self, args, annotaion_path, transform=None):
        self.sfew_path = args.sfew_path
        self.transform = transform

        name_c = 0
        label_c = 1
        dataset = pd.read_csv(annotaion_path, sep=' ', header=None)
        # dataset = df[df[name_c].str.startswith('train')]
        print(dataset.groupby([1]).size())

        # notice the RAF-DB label starts from 1 while label of other dataset starts from 0:
        self.label = dataset.iloc[:, label_c].values
        images_names = dataset.iloc[:, name_c].values
        self.image_paths = []

        for f in images_names:
            file_name = os.path.join(self.sfew_path, f)
            self.image_paths.append(file_name)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        label = self.label[idx]
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image)
        return image, label, idx


class AffectDataset(torch.utils.data.Dataset):
    """
      # #################### AffecNet EmoLabel (8 categories) ################
        # 0: Neutral,
        # 1: Happiness,
        # 2: Sadness,
        # 3: Surprise,
        # 4: Fear,
        # 5: Disgust,
        # 6: Anger,
        # 7: Contempt,
        # others excluding 8: None, # 9: Uncertain, # 10: No-Face
    """

    def __init__(self, data_path, annotaion_path, transform=None, is_train=True):
        # self.affectnet_path = args.affectnet_path
        self.affectnet_path = data_path
        self.transform = transform

        name_c = 0
        label_c = 1
        dataset = pd.read_csv(annotaion_path, sep=' ', header=None)
        # dataset = df[df[name_c].str.startswith('train')]
        print(dataset.groupby([1]).size())

        self.label = dataset.iloc[:, label_c].values
        images_names = dataset.iloc[:, name_c].values
        self.image_paths = []

        for f in images_names:
            file_name = os.path.join(self.affectnet_path, f)
            self.image_paths.append(file_name)
        self.is_train = is_train
    def __len__(self):
        return len(self.image_paths)
    def get_labels(self):
        return self.label
    def __getitem__(self, idx):
        label = self.label[idx]
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image)
        return image, label, idx, {}



class FplusDataSet(torch.utils.data.Dataset):
    """
     # #################### FERPlus EmoLabel (8 categories) ################
        0: neutral,
        1: happiness,
        2: surprise,
        3: sadness,
        4: anger,
        5: disgust,
        6: fear,
        7: contempt
    """

    def __init__(self, args, annotaion_path, transform=None):
        self.fplus_path = args.fplus_path
        self.transform = transform

        name_c = 0
        label_c = 1
        dataset = pd.read_csv(annotaion_path, sep=' ', header=None)
        # dataset = df[df[name_c].str.startswith('train')]
        print(dataset.groupby([1]).size())
        self.label = dataset.iloc[:, label_c].values
        images_names = dataset.iloc[:, name_c].values
        self.image_paths = []

        for f in images_names:
            file_name = os.path.join(self.fplus_path, f)
            self.image_paths.append(file_name)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        label = self.label[idx]
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image)
        return image, label, idx


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):

        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        label_to_count = {}  # sample numbers for each emotion category
        for idx in self.indices:
            label = self._get_label(dataset, idx)

            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]  # weighted by sample numbers
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)

        if dataset_type in [AffectDataset, SFEW2Dataset]:
            return dataset.label[idx]
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class DatasetFromSampler(Dataset):
    """
    # ##################### Copied from catalyst #####################
    # https://github.com/catalyst-team/catalyst/blob/e99f90655d0efcf22559a46e928f0f98c9807ebf/catalyst/data/sampler.py#L499

    Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """
    # ##################### Copied from catalyst #####################
    # https://github.com/catalyst-team/catalyst/blob/e99f90655d0efcf22559a46e928f0f98c9807ebf/catalyst/data/sampler.py#L499

    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.
        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


# def debug(show_image=False):
#     from time import time
#     import argparse
#     from randaugument import RandomAugment

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--raf_path', type=str, default='data/images_labels/link2RAF-DB/basic', help='raf_dataset_path')
#     parser.add_argument('--train_label_path', type=str, default='data/images_labels/link2RAF-DB/basic/EmoLabel'
#                                                                 '/train_label.txt',
#                         help='label_path')
#     parser.add_argument('--test_label_path', type=str, default='data/images_labels/link2RAF-DB/basic/EmoLabel'
#                                                                '/test_label.txt',
#                         help='label_path')

#     args = parser.parse_args()

#     trans_weak = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.RandomApply([transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25),
#                                 transforms.RandomAffine(degrees=0, translate=(.1, .1),
#                                                         scale=(1.0, 1.25),
#                                                         resample=Image.BILINEAR)], p=0.5),

#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225]),
#         transforms.RandomErasing(scale=(0.02, 0.25))
#     ])

#     trans_strong = transforms.Compose([
#         my_transforms.Resize((224, 224)),
#         my_transforms.PadandRandomCrop(border=4, cropsize=(224, 224)),
#         my_transforms.RandomHorizontalFlip(p=0.5),
#         RandomAugment(2, 10),  #
#         my_transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225]),
#         my_transforms.ToTensor(),
#     ])

#     data_transforms_val = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])])

#     train_dataset = RafDataset(args, transform=trans_weak, basic_aug=True)
#     print('Train set size:', train_dataset.__len__())

#     test_dataset = RafDataset(args, transform=data_transforms_val)
#     print('Validation set size:', test_dataset.__len__())

#     start = time()
#     batch = 0
#     for index in range(train_dataset.__len__()):
#         batch += 1
#         image, labels, indexes = [v for v in
#                                          train_dataset.__getitem__(index)]

#         if show_image:
#             plt.imshow(image.numpy().transpose((1, 2, 0)))
#             plt.show()

#     print("produce %d samples per second: " % (batch / (time() - start)))  # produce 115 samples per second


if __name__ == '__main__':
    #  ############ test timm package  ############
    import timm
    import ssl
    import urllib
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from models import vit

    model = timm.create_model('myvit_base_patch16_224', pretrained=True)

    ssl._create_default_https_context = ssl._create_unverified_context

    # # ############ test the data generator  ############
    import timeit
    print(timeit.timeit(stmt='debug_ferplus(True)',
                        setup="from __main__ import debug_ferplus;",
                        number=1))  # run for number times

    # # ################# test timm API  #################
    # model = timm.create_model('vit_base_patch16_224', pretrained=True)
    # model.eval()
    # # fc = model.get_classifier()  # return model.head
    #
    # config = resolve_data_config({}, model=model)
    # transform = create_transform(**config)
    #
    # img = Image.open("data/dog.jpg").convert('RGB')
    # tensor = transform(img).unsqueeze(0)  # transform and add batch dimension
    #
    # with torch.no_grad():
    #     out = model(tensor)
    # probabilities = torch.nn.functional.softmax(out[0], dim=0)
    # print(probabilities.shape)
    # # prints: torch.Size([1000])
    # # Get imagenet class mappings
    # url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
    # urllib.request.urlretrieve(url, filename)
    # with open("imagenet_classes.txt", "r") as f:
    #     categories = [s.strip() for s in f.readlines()]
    #
    # # Print top categories per image
    # top5_prob, top5_catid = torch.topk(probabilities, 5)
    # for i in range(top5_prob.size(0)):
    #     print(categories[top5_catid[i]], top5_prob[i].item())
