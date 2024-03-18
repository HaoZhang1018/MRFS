import os
import cv2
import torch
import numpy as np
from torch.utils import data
import random

from utils.transforms import generate_random_crop_pos, random_crop_pad_to_shape, normalize

def random_mirror(rgb, gt, modal_x):
    if random.random() >= 0.5:
        rgb = cv2.flip(rgb, 1)
        gt = cv2.flip(gt, 1)
        modal_x = cv2.flip(modal_x, 1)

    return rgb, gt, modal_x

def random_scale(rgb, gt, modal_x, scales):
    scale = random.choice(scales)
    sh = int(rgb.shape[0] * scale)
    sw = int(rgb.shape[1] * scale)
    rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
    modal_x = cv2.resize(modal_x, (sw, sh), interpolation=cv2.INTER_LINEAR)

    return rgb, gt, modal_x, scale

class TrainPre(object):
    def __init__(self, image_height, image_width, train_scale_array):
        self.image_height = image_height
        self.image_width = image_width
        self.train_scale_array = train_scale_array

    def __call__(self, rgb, gt, modal_x):
        rgb, gt, modal_x = random_mirror(rgb, gt, modal_x)
        if self.train_scale_array is not None:
            rgb, gt, modal_x, scale = random_scale(rgb, gt, modal_x, self.train_scale_array)

        rgb = normalize(rgb)
        modal_x = normalize(modal_x)

        crop_size = (self.image_height, self.image_width)
        crop_pos = generate_random_crop_pos(rgb.shape[:2], crop_size)

        p_rgb, Margin = random_crop_pad_to_shape(rgb, crop_pos, crop_size, 0)
        p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)
        p_modal_x, _ = random_crop_pad_to_shape(modal_x, crop_pos, crop_size, 0)

        p_rgb = p_rgb.transpose(2, 0, 1)
        p_modal_x = p_modal_x.transpose(2, 0, 1)
        
        Mask = np.zeros(p_rgb.shape)
        Mask[:, Margin[0]:(crop_size[0]-Margin[1]), Margin[2]:(crop_size[1]-Margin[3])] = 1.
        
        return p_rgb, p_gt, p_modal_x, Mask.astype(np.float32)

class ValPre(object):
    def __call__(self, rgb, gt, modal_x):
        return rgb, gt, modal_x

def get_train_loader(config, engine, dataset):
    data_setting = {'rgb_root': os.path.join(config.dataset_path, config.rgb_folder),
                    'rgb_format': config.rgb_format,
                    'x_root': os.path.join(config.dataset_path, config.x_folder),
                    'x_format': config.x_format,
                    'x_single_channel': config.x_is_single_channel,
                    'gt_root': os.path.join(config.dataset_path, config.label_folder),
                    'gt_format': config.label_format,
                    'transform_gt': config.gt_transform,
                    'class_names': config.class_names,
                    'train_source': os.path.join(config.dataset_path, "train.txt"),
                    'eval_source': os.path.join(config.dataset_path, "test.txt"),
                    }
    train_preprocess = TrainPre(config.image_height,
                                config.image_width,
                                config.train_scale_array,
                                )
    file_length = (config.num_train_imgs // config.batch_size + 1) * config.batch_size
    train_dataset = dataset(data_setting, "train", train_preprocess, file_length)

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=train_sampler)

    return train_loader, train_sampler

def get_val_loader(config, engine, dataset):
    data_setting = {'rgb_root': os.path.join(config.dataset_path, config.rgb_folder),
                    'rgb_format': config.rgb_format,
                    'x_root': os.path.join(config.dataset_path, config.x_folder),
                    'x_format': config.x_format,
                    'x_single_channel': config.x_is_single_channel,
                    'gt_root': os.path.join(config.dataset_path, config.label_folder),
                    'gt_format': config.label_format,
                    'transform_gt': config.gt_transform,
                    'class_names': config.class_names,
                    'train_source': os.path.join(config.dataset_path, "train.txt"),
                    'eval_source': os.path.join(config.dataset_path, "test.txt"),
                    }

    class ValPre(object):
        def __call__(self, rgb, gt, modal_x):
            rgb = normalize(rgb)
            modal_x = normalize(modal_x)
            rgb = rgb.transpose(2, 0, 1)
            modal_x = modal_x.transpose(2, 0, 1)
            rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()
            gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
            modal_x = torch.from_numpy(np.ascontiguousarray(modal_x)).float()
            return rgb, gt, modal_x

    val_pre = ValPre()

    val_sampler = None
    is_shuffle = False
    batch_size = 1

    val_dataset = dataset(data_setting, 'val', val_pre)

    if engine.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    val_loader = data.DataLoader(val_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=False,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=val_sampler)
    return val_loader, val_sampler