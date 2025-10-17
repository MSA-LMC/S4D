import sys
import os
startup_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, startup_dir)

import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
from functools import partial
from pathlib import Path
from collections import OrderedDict
import random
from unilearn.engine_for_finetuning import train_one_epoch, validation_one_epoch, sfer_validation_one_epoch, final_test, merge


from utils.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from utils.optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner
import torch.multiprocessing as mp

from datasets import build_dataset
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import multiple_samples_collate
import utils as utils
import unilearn.videomae


from data.torch_dataset import TorchDataset
from data.concat_dataset import ConcatDataset
from data.transforms.cutmixup import CutMixUp
from data.api import DefaultOmnivoreCollator
from sfer import build_dataset as build_sfer_dataset



def get_args():
    parser = argparse.ArgumentParser(
        'VideoMAE fine-tuning and evaluation script for video classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=100, type=int)

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--tubelet_size', type=int, default=2)
    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size')
    # mixture of experts
    parser.add_argument('--moe_layers', type=int,  default=6,
                        help='moe layers')
    parser.add_argument('--moe_type', default=None, choices=['sparse_moae','moe_lora','moe_adapters','moe_adapters2','MTL','sparse', 'sparse_with_capacity', 'simple'],
                        help='moe type')
    parser.add_argument('--num_experts', default=1, type=int,
                        help='num of experts')
    parser.add_argument('--lora_rank', default=16, type=int,
                    help='rank of LoRA')
    parser.add_argument('--lora_alpha', default=16, type=int,
                        help='alpha of LoRA')
    parser.add_argument('--top_k', default=1, type=int,
                        help='top K experts')
    parser.add_argument('--capacity_factor', default=1.0, type=float,
                        help='capacity factor')
    parser.add_argument('--aux_loss_coeff', type=float, default=1e-3,
        help='auxiliary loss coefficient for moe')


    parser.add_argument('--disable_eval_during_finetuning',
                        action='store_true', default=False)
    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float,
                        default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu',
                        action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--layer_decay', type=float, default=0.75)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--num_sample', type=int, default=2,
                        help='Repeated_aug (default: 2)')
    parser.add_argument('--aa', type=str, default='rand-m7-n4-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m7-n4-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--augment', action='store_true',
                        default=True, help='Use data augment')
    parser.set_defaults(augment=True)
    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--short_side_size', type=int, default=224)
    parser.add_argument('--test_num_segment', type=int, default=5)
    parser.add_argument('--test_num_crop', type=int, default=3)

    # Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false',
                        dest='use_mean_pooling')
    parser.add_argument('--mode', default='joint', choices=['joint', 'sfer', 'dfer']
                        ,type=str, help='trainig mode')
    
    # Dataset parameters
    parser.add_argument('--data_path', default='/path/to/list_kinetics-400', type=str,
                        help='dataset path')
    parser.add_argument('--train_label_path', default='/path/to/list_kinetics-400/train.csv', type=str,
                        help='train label path')
    parser.add_argument('--test_label_path', default='/path/to/list_kinetics-400/test.csv', type=str,
                        help='test label path')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=400, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std',
                        default=True, action='store_true')
    parser.add_argument('--num_segments', type=int, default=1)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--sampling_rate', type=int, default=4)
    parser.add_argument('--data_set', default='Kinetics-400', choices=['Kinetics-400', 'SSV2', 'UCF101', 'HMDB51', 'image_folder',
                        'DFEW', 'FERV39k', 'MAFW', 'RAVDESS', 'CREMA-D', 'ENTERFACE', 'CELEBV-TEXT'],
                        type=str, help='dataset')
    parser.add_argument('--sfer_data_set', default='affectnet-7', choices=['affectnet-7', 'affectnet-8', 'RAF-DB'],
                        type=str, help='sfer dataset')
    parser.add_argument('--sfer_data_rate', default=1.0,  type=float)
    parser.add_argument('--dfer_data_rate', default=1.0,  type=float)

    parser.add_argument('--oversamplering', action='store_true', default=False, help='oversamplering for imbalance dataset')
    
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume',
                        action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument(
        '--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--enable_deepspeed',
                        action='store_true', default=False)

    parser.add_argument('--val_metric', type=str, default='acc1', choices=['acc1', 'acc5', 'war', 'uar', 'weighted_f1', 'micro_f1', 'macro_f1'],
                        help='validation metric for saving best ckpt')
    parser.add_argument('--depth', default=None, type=int,
                        help='specify model depth, NOTE: only works when no_depth model is used!')

    parser.add_argument('--save_feature', action='store_true', default=False)
    parser.add_argument('--use_3d_face', action='store_true', default=False)

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init


def main(local_rank, nprocs, args, ds_init):
    args.local_rank = local_rank
    args.world_size = nprocs

    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)
    if args.sfer_data_set.startswith('affectnet'):
        args.oversamplering = True
        print(f"Using oversamplering for {args.sfer_data_set} dataset")
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(
        is_train=True, test_mode=False, args=args)
    sfer_dataset_train, sfer_nb_classes = build_sfer_dataset(
        is_train=True, test_mode=False, args=args)
            
    if args.disable_eval_during_finetuning:
        dataset_val = sfer_dataset_val = None
    else:
        dataset_val, _ = build_dataset(
            is_train=False, test_mode=False, args=args)
        sfer_dataset_val, _ = build_sfer_dataset(
            is_train=False, test_mode=False, args=args)


    dataset_test, _ = build_dataset(is_train=False, test_mode=True, args=args)
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    # sampler_train = torch.utils.data.DistributedSampler(
    #     dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    # )
    # print("Sampler_train = %s" % str(sampler_train))
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                  'This will slightly alter validation results as extra duplicate entries are added to achieve '
                  'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        
        sfer_sampler_val = torch.utils.data.DistributedSampler(
            sfer_dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    if args.num_sample > 1:
        collate_func = partial(multiple_samples_collate, fold=False)
    else:
        collate_func = None

    # data_loader_train = torch.utils.data.DataLoader(
    #     dataset_train, sampler=sampler_train,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_mem,
    #     drop_last=True,
    #     collate_fn=collate_func,
    # )

    data_video = TorchDataset(
        dataset=dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=DefaultOmnivoreCollator(output_key='video')
    )

    # for image
    if sfer_dataset_train:
        batch_transforms=CutMixUp(
        mixup_alpha=args.mixup, 
        cutmix_alpha=args.cutmix, 
        prob=args.mixup_prob,
        switch_prob=args.mixup_switch_prob,
        mode=args.mixup_mode,
        correct_lam = True,
        label_smoothing=args.smoothing,
        num_classes=sfer_nb_classes)

    data_image = TorchDataset(
            dataset=sfer_dataset_train,
            batch_size=args.batch_size * 2,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=args.pin_mem,
            drop_last=True,
            collate_fn=DefaultOmnivoreCollator(
                output_key='image',
                batch_transforms=batch_transforms),
            oversamplering=args.oversamplering #for imbalance dataset
    )
    
    if args.mode=='joint':
        data_loader_train = ConcatDataset(
                datasets=[data_video, data_image],
                max_steps='sum',
                repeat_factors=[args.dfer_data_rate, args.sfer_data_rate]
            )
    elif args.mode=='sfer':
        data_loader_train = ConcatDataset(
                datasets=[data_image],
                max_steps='sum',
                repeat_factors=[args.sfer_data_rate]
            )
        dataset_val=None 
        dataset_test=None 

    elif args.mode=='dfer':
        data_loader_train = data_video
        sfer_dataset_val=None

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_val = None

    if sfer_dataset_val is not None:
        sfer_data_loader_val = torch.utils.data.DataLoader(
            sfer_dataset_val, sampler=sfer_sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        sfer_data_loader_val = None

    if dataset_test is not None:
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_test = None

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    args.moe_layers = list(range(args.moe_layers, 12))
    print(f'model: {args.model}, moe_type: {args.moe_type}, moe_layers: {args.moe_layers}, num_experts: {args.num_experts}, top_k: {args.top_k}, capacity_factor: {args.capacity_factor}, lora_rank: {args.lora_rank}, lora_alpha: {args.lora_alpha}')
    if 'moe' in args.model:
        model = create_model(
            args.model,
            pretrained=args.finetune,
            num_classes=args.nb_classes,
            multi_head=[sfer_nb_classes, args.nb_classes],
            moe={
                'type': args.moe_type,
                'layers':args.moe_layers,
                'num_experts': args.num_experts,
                'top_k': args.top_k,
                'capacity_factor': args.capacity_factor,
                'lora_rank': args.lora_rank,
                'lora_alpha': args.lora_alpha,
            } 
        )
    else:
        model = create_model(
            args.model,
            pretrained=args.finetune,
            num_classes=args.nb_classes,
            multi_head=[sfer_nb_classes, args.nb_classes],
        )

    # patch_size = model.patch_embed.patch_size
    # print("Patch size = %s" % str(patch_size))
    # args.window_size = (args.num_frames // 2, args.input_size //
    #                     patch_size[0], args.input_size // patch_size[1])
    # args.patch_size = patch_size

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            print(args.finetune)
            checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith('backbone.'):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith('encoder.'):
                new_dict[key[8:]] = checkpoint_model[key]
            elif key.startswith('head.video'):
                mew_key = key.replace('head.video', 'head')
                new_dict[mew_key] = checkpoint_model[key]
            else:
                new_dict[key] = checkpoint_model[key]
        for key in all_keys:
            if key.startswith('encoder.'):
                try:
                    layer = int(key.split('.')[2])
                    if key.startswith('encoder.blocks') and layer in args.moe_layers and f'blocks.{layer}.mlp' in key:
                        for expert in range(args.num_experts):
                            new_key = key[8:].replace(f'mlp', f'mlp.experts.{expert}')
                            new_dict[new_key] = checkpoint_model[key]
                    else:
                        new_dict[key[8:]] = checkpoint_model[key]
                except:
                    new_dict[key[8:]] = checkpoint_model[key]
        
        checkpoint_model = new_dict

        # # interpolate position embedding
        # if 'pos_embed' in checkpoint_model:
        #     pos_embed_checkpoint = checkpoint_model['pos_embed']
        #     embedding_size = pos_embed_checkpoint.shape[-1]  # channel dim
        #     num_patches = model.patch_embed.num_patches
        #     num_extra_tokens = model.pos_embed.shape[-2] - num_patches  # 0/1

        #     # height (== width) for the checkpoint position embedding
        #     orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens)//(
        #         args.num_frames // model.patch_embed.tubelet_size)) ** 0.5)
        #     # height (== width) for the new position embedding
        #     new_size = int(
        #         (num_patches // (args.num_frames // model.patch_embed.tubelet_size)) ** 0.5)
        #     # class_token and dist_token are kept unchanged
        #     if orig_size != new_size:
        #         print("Position interpolate from %dx%d to %dx%d" %
        #               (orig_size, orig_size, new_size, new_size))
        #         extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        #         # only the position tokens are interpolated
        #         pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        #         # B, L, C -> BT, H, W, C -> BT, C, H, W
        #         pos_tokens = pos_tokens.reshape(
        #             -1, args.num_frames // model.patch_embed.tubelet_size, orig_size, orig_size, embedding_size)
        #         pos_tokens = pos_tokens.reshape(-1, orig_size,
        #                                         orig_size, embedding_size).permute(0, 3, 1, 2)
        #         pos_tokens = torch.nn.functional.interpolate(
        #             pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        #         # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
        #         pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(
        #             -1, args.num_frames // model.patch_embed.tubelet_size, new_size, new_size, embedding_size)
        #         pos_tokens = pos_tokens.flatten(1, 3)  # B, L, C
        #         new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        #         checkpoint_model['pos_embed'] = new_pos_embed

        utils.load_state_dict(model, checkpoint_model,
                              prefix=args.model_prefix)

    model.to(device)

    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print(f'number of params: {n_parameters / 1e6} M')

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    # num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    num_training_steps_per_epoch = len(data_loader_train.get_loader(epoch=0)) // args.update_freq

    args.lr = args.lr * total_batch_size / 8
    args.min_lr = args.min_lr * total_batch_size / 256
    args.warmup_lr = args.warmup_lr * total_batch_size / 256
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" %
          num_training_steps_per_epoch)

    if args.layer_decay < 1.0:
        num_layers = model_without_ddp.get_num_layers()
        assigner = LayerDecayValueAssigner(
            list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))
    skip_weight_decay_list = model.no_weight_decay()
    try:
        skip_weight_decay_list = model.no_weight_decay()
    except:
        skip_weight_decay_list = None
    print("Skip weight decay list: ", skip_weight_decay_list)

    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None)
        model, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=optimizer_params, dist_init_required=not args.distributed,
        )

        print("model.gradient_accumulation_steps() = %d" %
              model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module

        optimizer = create_optimizer(
            args, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None,
            get_layer_scale=assigner.get_scale if assigner is not None else None)
        loss_scaler = NativeScaler()

    print("Use step level LR scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" %
          (max(wd_schedule_values), min(wd_schedule_values)))

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema) #TODO: Resume best metric

    if args.eval:
        preds_file = os.path.join(args.output_dir, str(global_rank) + '.txt')
        test_stats = final_test(
            data_loader_test, model, device, preds_file, save_feature=args.save_feature)
        torch.distributed.barrier()
        if global_rank == 0:
            print("Start merging results...")
            final_top1, final_top5, pred_dict = merge(
                args.output_dir, num_tasks, args)
            print(
                f"Accuracy of the network on the {len(dataset_test)} test videos: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%")
            log_stats = {'Final Top-1': final_top1,
                         'Final Top-5': final_top5}
            # me: more metrics
            from sklearn.metrics import confusion_matrix, f1_score
            preds, labels = pred_dict['pred'], pred_dict['label']
            print(f'Total test samples: {len(preds)}')
            conf_mat = confusion_matrix(y_pred=preds, y_true=labels)
            print(f'Confusion Matrix:\n{conf_mat}')
            class_acc = conf_mat.diagonal() / conf_mat.sum(axis=1)
            print(f"Class Accuracies: {[f'{i:.2%}' for i in class_acc]}")
            uar = np.mean(class_acc)
            war = conf_mat.trace() / conf_mat.sum()
            print(f'UAR: {uar:.2%}, WAR: {war:.2%}')
            weighted_f1 = f1_score(
                y_pred=preds, y_true=labels, average='weighted')
            micro_f1 = f1_score(y_pred=preds, y_true=labels, average='micro')
            macro_f1 = f1_score(y_pred=preds, y_true=labels, average='macro')
            print(
                f'Weighted F1: {weighted_f1:.4f}, micro F1: {micro_f1:.4f}, macro F1: {macro_f1:.4f}')

            if args.output_dir and utils.is_main_process():
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
                    f.write(f'Final UAR: {uar:.2%}, Final WAR: {war:.2%}\n')
                    f.write(f'Final Confusion Matrix:\n{conf_mat}\n')
                    f.write(
                        f'Final Class Accuracies: {[f"{i:.2%}" for i in class_acc]}\n')
                    f.write(
                        f'Final Weighted F1: {weighted_f1:.4f}, Final Micro F1: {micro_f1:.4f}, Final Macro F1: {macro_f1:.4f}\n')

                import pandas as pd
                df = pd.DataFrame(pred_dict)
                df.to_csv(os.path.join(
                    args.output_dir, 'pred.csv'), index=False)

        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_metric = -1e8 if args.val_metric not in ['loss'] else 1e8
    sfer_best_metric = -1e8 if args.val_metric not in ['loss'] else 1e8

    best_epoch, sfer_best_epoch = None, None
    for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed:
        #     data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(
                epoch * num_training_steps_per_epoch * args.update_freq)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train.get_loader(epoch=epoch), optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
            log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch, aux_loss_coeff=args.aux_loss_coeff,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
        )
        if args.output_dir and args.save_ckpt:
            # if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
            utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)
        test_stats=None
        if data_loader_val is not None:
            test_stats = validation_one_epoch(data_loader_val, model, device)
            print(
                f"Accuracy of the network on the {len(dataset_val)} val videos: {test_stats['acc1']:.1f}%")
            if (args.val_metric not in ['loss'] and best_metric < test_stats[args.val_metric]) or \
                    (args.val_metric in ['loss'] and best_metric > test_stats[args.val_metric]):
                best_metric = test_stats[args.val_metric]
                best_epoch = epoch
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema, is_best=True)

            print(
                f"Best '{args.val_metric.upper()}': {best_metric:.4f}% (epoch={best_epoch})")
            if log_writer is not None:
                log_writer.update(
                    val_acc1=test_stats['acc1'], head="perf", step=epoch)
                log_writer.update(
                    val_acc5=test_stats['acc5'], head="perf", step=epoch)
                log_writer.update(
                    val_loss=test_stats['loss'], head="perf", step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
            
        if sfer_data_loader_val is not None:
            sfer_test_stats = sfer_validation_one_epoch(sfer_data_loader_val, model, device)
            print(
                f"Accuracy of the network on the {len(sfer_dataset_val)} val images: {sfer_test_stats['sfer_acc1']:.2f}%")
            if (args.val_metric not in ['loss'] and sfer_best_metric < sfer_test_stats['sfer_acc1']) or \
                    (args.val_metric in ['loss'] and sfer_best_metric > sfer_test_stats['sfer_acc1']):
                sfer_best_metric = sfer_test_stats['sfer_acc1']
                sfer_best_epoch = epoch
                if args.output_dir and args.save_ckpt and args.mode=='sfer':
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema, is_best=True)
            print(
                f"Best SFER '{args.val_metric.upper()}': {sfer_best_metric:.4f}% (epoch={sfer_best_epoch})")
            if log_writer is not None:
                log_writer.update(
                    val_acc1=sfer_test_stats['sfer_acc1'], head="perf", step=epoch)
                log_writer.update(
                    val_acc5=sfer_test_stats['sfer_acc5'], head="perf", step=epoch)
                log_writer.update(
                    val_loss=sfer_test_stats['sfer_loss'], head="perf", step=epoch)
            #update log stats to include sfer stats
            if test_stats is None:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'sfer_val_{k}': v for k, v in sfer_test_stats.items()},
                             'epoch': epoch,
                             'n_parameters': n_parameters}
            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'val_{k}': v for k, v in test_stats.items()},
                            **{f'sfer_val_{k}': v for k, v in sfer_test_stats.items()},
                            'epoch': epoch,
                            'n_parameters': n_parameters}
                
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        remain_time = (time.time() - start_time) / (epoch - args.start_epoch +1) * (args.epochs - epoch) 
        remain_time_str = str(datetime.timedelta(seconds=int(remain_time)))
        print('Training remain time {}'.format(remain_time_str))

    preds_file = os.path.join(args.output_dir, str(global_rank) + '.txt')
    test_stats = final_test(data_loader_test, model,
                            device, preds_file, save_feature=args.save_feature)

    torch.distributed.barrier()

    if global_rank == 0:
        print("Start merging results...")
        # me: original merge
        final_top1, final_top5, pred_dict = merge(
            args.output_dir, num_tasks, args)
        print(
            f"Accuracy of the network on the {len(dataset_test)} test videos using last epoch model: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%")
        log_stats = {'Final Top-1 (last epoch)': final_top1,
                     'Final Top-5 (last epoch)': final_top5}
        # me: more metrics
        from sklearn.metrics import confusion_matrix, f1_score
        preds, labels = pred_dict['pred'], pred_dict['label']
        print(f'Total test samples: {len(preds)}')
        conf_mat = confusion_matrix(y_pred=preds, y_true=labels)
        print(f'Confusion Matrix:\n{conf_mat}')
        class_acc = conf_mat.diagonal() / conf_mat.sum(axis=1)
        print(f"Class Accuracies: {[f'{i:.2%}' for i in class_acc]}")
        uar = np.mean(class_acc)
        war = conf_mat.trace() / conf_mat.sum()
        print(f'UAR: {uar:.2%}, WAR: {war:.2%}')
        weighted_f1 = f1_score(y_pred=preds, y_true=labels, average='weighted')
        micro_f1 = f1_score(y_pred=preds, y_true=labels, average='micro')
        macro_f1 = f1_score(y_pred=preds, y_true=labels, average='macro')
        print(
            f'Weighted F1: {weighted_f1:.4f}, micro F1: {micro_f1:.4f}, macro F1: {macro_f1:.4f}')

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                # last
                f.write(f"Evaluation on the test set using last epoch model:\n")
                f.write(json.dumps(log_stats) + "\n")
                # me: save to log.txt
                f.write(f'Final UAR: {uar:.2%}, Final WAR: {war:.2%}\n')
                f.write(f'Final Confusion Matrix:\n{conf_mat}\n')
                f.write(
                    f'Final Class Accuracies: {[f"{i:.2%}" for i in class_acc]}\n')
                f.write(
                    f'Final Weighted F1: {weighted_f1:.4f}, Final Micro F1: {micro_f1:.4f}, Final Macro F1: {macro_f1:.4f}\n')
            # me: save preds and labels
            import pandas as pd
            # last
            df = pd.DataFrame(pred_dict)
            df.to_csv(os.path.join(args.output_dir, 'pred.csv'), index=False)

    print('Final test on the best epoch model')
    checkpoint = torch.load(os.path.join(
        args.output_dir, 'checkpoint-best.pth'), map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'])

    preds_file = os.path.join(args.output_dir, str(global_rank) + '.txt')
    test_stats = final_test(data_loader_test, model,
                            device, preds_file, save_feature=args.save_feature)

    torch.distributed.barrier()

    if global_rank == 0:
        print("Start merging results...")
        # me: original merge
        final_top1, final_top5, pred_dict = merge(
            args.output_dir, num_tasks, args)
        print(
            f"Accuracy of the network on the {len(dataset_test)} test videos using best epoch model: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%")
        log_stats = {'Final Top-1 (best epoch)': final_top1,
                     'Final Top-5 (best epoch)': final_top5}
        # me: more metrics
        from sklearn.metrics import confusion_matrix, f1_score
        preds, labels = pred_dict['pred'], pred_dict['label']
        print(f'Total test samples: {len(preds)}')
        conf_mat = confusion_matrix(y_pred=preds, y_true=labels)
        print(f'Confusion Matrix:\n{conf_mat}')
        class_acc = conf_mat.diagonal() / conf_mat.sum(axis=1)
        print(f"Class Accuracies: {[f'{i:.2%}' for i in class_acc]}")
        uar = np.mean(class_acc)
        war = conf_mat.trace() / conf_mat.sum()
        print(f'UAR: {uar:.2%}, WAR: {war:.2%}')
        weighted_f1 = f1_score(y_pred=preds, y_true=labels, average='weighted')
        micro_f1 = f1_score(y_pred=preds, y_true=labels, average='micro')
        macro_f1 = f1_score(y_pred=preds, y_true=labels, average='macro')
        print(
            f'Weighted F1: {weighted_f1:.4f}, micro F1: {micro_f1:.4f}, macro F1: {macro_f1:.4f}')

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                # last
                f.write(f"Evaluation on the test set using best epoch model:\n")
                f.write(json.dumps(log_stats) + "\n")
                # me: save to log.txt
                f.write(f'Final UAR: {uar:.2%}, Final WAR: {war:.2%}\n')
                f.write(f'Final Confusion Matrix:\n{conf_mat}\n')
                f.write(
                    f'Final Class Accuracies: {[f"{i:.2%}" for i in class_acc]}\n')
                f.write(
                    f'Final Weighted F1: {weighted_f1:.4f}, Final Micro F1: {micro_f1:.4f}, Final Macro F1: {macro_f1:.4f}\n')
            # me: save preds and labels
            import pandas as pd
            # last
            df = pd.DataFrame(pred_dict)
            df.to_csv(os.path.join(args.output_dir, 'pred.csv'), index=False)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts, ds_init = get_args()
    print(opts.train_label_path, opts.test_label_path, opts.save_ckpt_freq)
    opts.port = random.randint(10000, 20000)
    opts.nprocs = torch.cuda.device_count()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    
    mp.spawn(main, nprocs=opts.nprocs, args=(opts.nprocs, opts, ds_init))
    
