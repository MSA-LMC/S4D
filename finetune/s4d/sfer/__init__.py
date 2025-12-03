from .dataset import *
import torchvision

from torchvision.transforms import InterpolationMode
from .rand_auto_aug import RandAugment
# data_augment = torchvision.transforms.Compose([
#     torchvision.transforms.ToPILImage(),
#     torchvision.transforms.RandomHorizontalFlip(p=0.5),
#     torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(brightness=0.25, contrast=0.25,
#                                                                            saturation=0.25, hue=0.25),
#                                         # 仿射变换对训练准确率有显著影响，但是不能太剧烈，以下设置效果就挺好
#                                         torchvision.transforms.RandomAffine(degrees=15, translate=(.1, .1),
#                                                                             scale=(1.0, 1.25),
#                                                                             interpolation=InterpolationMode.BICUBIC)],
#                                                                             p=0.5),

#     torchvision.transforms.Resize((160,160)),  # original size: 224*224
#     torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                          std=[0.229, 0.224, 0.225]),
#     torchvision.transforms.RandomErasing(
#         scale=(0.02, 0.25)),  # beneficial to final accuracy
# ])
data_augment = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize((192, 192)),
    torchvision.transforms.RandomCrop((160, 160)),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    RandAugment(magnitude=9, magnitude_std=0.5, increasing_severity=True),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.RandomErasing(p=0.25),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
])
data_transforms_val = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize((160, 160)),  # original size: 224*224
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std = [0.229, 0.224, 0.225])
])


def build_dataset(is_train=True, test_mode=False, args=None):
    data_type=args.sfer_data_set
    print(f"using {data_type} dataset")
    if data_type=='affectnet-8':
        if is_train:
            return AffectDataset(
                data_path="AffectNetdataset/Manually_Annotated_Images",
                annotaion_path="data/AffectNetdataset/train_affectnet_2w.txt",
                transform=data_augment,
                is_train=is_train
            ), 8
        else:
            return AffectDataset(
                data_path="data/AffectNetdataset/Manually_Annotated_Images",
                annotaion_path="data/data/AffectNetdataset/test_affectnet7_2w.txt",
                transform=data_transforms_val,
                is_train=is_train
            ), 8
    elif data_type=='affectnet-7':
        if is_train:
            return AffectDataset(
                data_path="data/AffectNetdataset/Manually_Annotated_Images",
                annotaion_path="data/AffectNetdataset/train_affectnet7_2w.txt",
                transform=data_augment,
                is_train=True
            ), 7
        else:
            return AffectDataset(
                data_path="data/AffectNetdataset/Manually_Annotated_Images",
                annotaion_path="data/AffectNetdataset/test_affectnet7_2w.txt",
                transform=data_transforms_val,
                is_train=False
            ), 7
    else:
        print('Not support yet')
