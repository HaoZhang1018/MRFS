import os
import argparse
import numpy as np

import torch
import torch.nn as nn

from utils.pyt_utils import ensure_dir, parse_devices
from engine.evaluator_vision import Evaluator
from engine.logger import get_logger
from dataloader.RGBXDataset import RGBXDataset
from models.model import MRFS
from dataloader.dataloader import ValPre
from PIL import Image

logger = get_logger()

class VisionEvaluator(Evaluator):

    def func_per_iteration_vision(self, data, device):
        img = data['data']
        modal_x = data['modal_x']
        name = data['fn']
        Fuse = self.sliding_eval_rgbX_vision(img, modal_x, self.eval_crop_size, self.eval_stride_rate, device)

        Fuse = (Fuse - Fuse.min()) / (Fuse.max() - Fuse.min()) * 255.0
        
        if self.save_path is not None:
            ensure_dir(self.save_path)

            fn = name + '.png'

            # save colored result
            result_img = Image.fromarray(Fuse.astype(np.uint8), mode='RGB')
            result_img.save(os.path.join(self.save_path, fn))


if __name__ == "__main__":
    dataset_name = "MFNet"
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='MRFS', type=str)
    parser.add_argument('-d', '--devices', default='0', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--save_path', '-p', default=f'./results_Fusion/{dataset_name}')
    # Dataset Config
    parser.add_argument('--dataset_path', default="./dataset/MFNet", type=str, help='absolute path of the dataset root')
    parser.add_argument('--rgb_folder', default="visible", type=str, help='folder for visible light images')
    parser.add_argument('--rgb_format', default=".png", type=str, help='the load format for visible light images')
    parser.add_argument('--x_folder', default="infrared", type=str, help='folder for thermal imaging images')
    parser.add_argument('--x_format', default=".png", type=str, help='the load format for thermal imaging images')
    parser.add_argument('--x_is_single_channel', default=True, type=bool,
                        help='True for raw depth, thermal and aolp/dolp(not aolp/dolp tri) input')
    parser.add_argument('--label_folder', default="label", type=str, help='folder for segmentation label image')
    parser.add_argument('--label_format', default=".png", type=str, help='the load format for segmentation label image')
    parser.add_argument('--gt_transform', default=False, type=bool, help='')
    parser.add_argument('--num_classes', default=9, type=int, help='')
    parser.add_argument('--class_names',
                        default=['unlabeled', 'car', 'person', 'bike', 'curve', 'car_stop', 'guardrail', 'color_cone', 'bump'],
                        type=list, help='the class names of all classes')
    # Network Config
    backbone = "mit_b4"
    parser.add_argument('--backbone', default=backbone, type=str, help='the backbone network to load')
    parser.add_argument('--decoder_embed_dim', default=512, type=int, help='')
    # Val Config
    parser.add_argument('--eval_crop_size', default=[480, 640], type=list, help='')
    parser.add_argument('--eval_stride_rate', default=2/3, type=float, help='')
    parser.add_argument('--eval_scale_array', default=[1], type=list, help='')
    parser.add_argument('--is_flip', default=False, type=bool, help='')
    log_dir = f"./checkpoints/log_{dataset_name}_{backbone}"
    parser.add_argument('--log_dir', default=log_dir, type=str, help=' ')
    parser.add_argument('--log_dir_link', default=log_dir, type=str, help='')
    parser.add_argument('--checkpoint_dir', default=os.path.join(log_dir, "weights"), type=str, help='')

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)

    network = MRFS(cfg=args, criterion=None, norm_layer=nn.BatchNorm2d)
    data_setting = {'rgb_root': os.path.join(args.dataset_path, args.rgb_folder),
                    'rgb_format': args.rgb_format,
                    'x_root': os.path.join(args.dataset_path, args.x_folder),
                    'x_format': args.x_format,
                    'x_single_channel': args.x_is_single_channel,
                    'gt_root': os.path.join(args.dataset_path, args.label_folder),
                    'gt_format': args.label_format,
                    'transform_gt': args.gt_transform,
                    'class_names': args.class_names,
                    'train_source': os.path.join(args.dataset_path, "train.txt"),
                    'eval_source': os.path.join(args.dataset_path, "test.txt")}
    val_pre = ValPre()
    dataset = RGBXDataset(data_setting, 'val', val_pre)
 
    with torch.no_grad():
        Fuser = VisionEvaluator(args, dataset, network, all_dev, args.verbose, args.save_path)
        Fuser.run(args.checkpoint_dir, args.epochs)