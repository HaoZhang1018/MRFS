import os
import cv2
import numpy as np
import time
from tqdm import tqdm
from timm.models.layers import to_2tuple

import torch
import multiprocessing as mp

from .logger import get_logger
from utils.pyt_utils import load_model
from utils.transforms import pad_image_to_shape, normalize

logger = get_logger()


class Evaluator(object):
    def __init__(self, config, dataset, network, devices, verbose=False, save_path=None):
        self.eval_time = 0
        self.dataset = dataset
        self.ndata = self.dataset.get_length()
        self.network = network
        self.eval_crop_size = config.eval_crop_size
        self.eval_stride_rate = config.eval_stride_rate
        self.class_num = config.num_classes
        self.multi_scales = config.eval_scale_array
        self.is_flip = config.is_flip
        self.devices = devices
        self.val_func = None
        self.context = mp.get_context('spawn')
        self.results_queue = self.context.Queue(self.ndata)
        self.verbose = verbose
        self.save_path = save_path

    def run(self, model_path, model_indice):
        """There are four evaluation modes:
            1.only eval a .pth model: -e *.pth
            2.only eval a certain epoch: -e epoch
            3.eval all epochs in a given section: -e start_epoch-end_epoch
            4.eval all epochs from a certain started epoch: -e start_epoch-
            """
        if '.pth' in model_indice:
            models = [model_indice, ]
        elif "-" in model_indice:
            start_epoch = int(model_indice.split("-")[0])
            end_epoch = model_indice.split("-")[1]

            models = os.listdir(model_path)
            models.remove("epoch-last.pth")
            sorted_models = [None] * len(models)
            model_idx = [0] * len(models)

            for idx, m in enumerate(models):
                num = m.split(".")[0].split("-")[1]
                model_idx[idx] = num
                sorted_models[idx] = m
            model_idx = np.array([int(i) for i in model_idx])

            down_bound = model_idx >= start_epoch
            up_bound = [True] * len(sorted_models)
            if end_epoch:
                end_epoch = int(end_epoch)
                assert start_epoch < end_epoch
                up_bound = model_idx <= end_epoch
            bound = up_bound * down_bound
            model_slice = np.array(sorted_models)[bound]
            models = [os.path.join(model_path, model) for model in
                      model_slice]
        else:
            if os.path.exists(model_path):
                models = [os.path.join(model_path, 'epoch-%s.pth' % model_indice), ]
            else:
                models = [None]

        for model in models:
            logger.info("Load Model: %s" % model)
            self.val_func = load_model(self.network, model)
            if len(self.devices ) == 1:
                self.single_process_evalutation_vision()
            else:
                self.multi_process_evaluation_vision()


    def single_process_evalutation_vision(self):
        start_eval_time = time.perf_counter()

        logger.info('GPU %s handle %d data.' % (self.devices[0], self.ndata))
        for idx in tqdm(range(self.ndata)):
            dd = self.dataset[idx]
            self.func_per_iteration_vision(dd,self.devices[0])
                    

    def func_per_iteration(self, data, device):
        raise NotImplementedError

    def compute_metric(self, results):
        raise NotImplementedError


    # add new funtion for rgb and modal X segmentation
    def sliding_eval_rgbX_vision(self, img, modal_x, crop_size, stride_rate, device=None):
        crop_size = to_2tuple(crop_size)
        ori_rows, ori_cols, _ = img.shape
        processed_pred = np.zeros((ori_rows, ori_cols, self.class_num))

        for s in self.multi_scales:
            img_scale = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            if len(modal_x.shape) == 2:
                modal_x_scale = cv2.resize(modal_x, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
            else:
                modal_x_scale = cv2.resize(modal_x, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)

            new_rows, new_cols, _ = img_scale.shape
            processed_pred = self.scale_process_rgbX_vision(img_scale, modal_x_scale, (ori_rows, ori_cols),
                                                        crop_size, stride_rate, device)
        return processed_pred

    def scale_process_rgbX_vision(self, img, modal_x, ori_shape, crop_size, stride_rate, device=None):
        new_rows, new_cols, c = img.shape
        long_size = new_cols if new_cols > new_rows else new_rows

        if new_cols <= crop_size[1] or new_rows <= crop_size[0]:
            input_data, input_modal_x, margin = self.process_image_rgbX_vision(img, modal_x, crop_size)
            score = self.val_func_process_rgbX_vision(input_data, input_modal_x, device) 

            score = score[:, margin[0]:(score.shape[1] - margin[1]), margin[2]:(score.shape[2] - margin[3])]

            
        score = score.permute(1, 2, 0)
        data_output = cv2.resize(score.cpu().numpy(), (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_LINEAR)

        return data_output

    def val_func_process_rgbX_vision(self, input_data, input_modal_x, device=None):
        input_data = np.ascontiguousarray(input_data[None, :, :, :], dtype=np.float32)
        input_data = torch.FloatTensor(input_data).cuda(device)
    
        input_modal_x = np.ascontiguousarray(input_modal_x[None, :, :, :], dtype=np.float32)
        input_modal_x = torch.FloatTensor(input_modal_x).cuda(device)
    
        with torch.cuda.device(input_data.get_device()):
            self.val_func.eval()
            self.val_func.to(input_data.get_device())
            with torch.no_grad():
                score, Fus_img = self.val_func(input_data, input_modal_x)
                Fus_img = Fus_img[0]

        return Fus_img

    def process_image_rgbX_vision(self, img, modal_x, crop_size=None):
        p_img = img
        p_modal_x = modal_x
    
        if img.shape[2] < 3:
            im_b = p_img
            im_g = p_img
            im_r = p_img
            p_img = np.concatenate((im_b, im_g, im_r), amodal_xis=2)
    
        p_img = normalize(p_img)
        if len(modal_x.shape) == 2:
            p_modal_x = normalize(p_modal_x)
        else:
            p_modal_x = normalize(p_modal_x)
    
        if crop_size is not None:
            p_img, margin = pad_image_to_shape(p_img, crop_size, cv2.BORDER_CONSTANT, value=0)
            p_modal_x, _ = pad_image_to_shape(p_modal_x, crop_size, cv2.BORDER_CONSTANT, value=0)

            p_img = p_img.transpose(2, 0, 1)
            if len(modal_x.shape) == 2:
                p_modal_x = p_modal_x[np.newaxis, ...]
            else:
                p_modal_x = p_modal_x.transpose(2, 0, 1) # 3 H W
        
            return p_img, p_modal_x, margin
    
        p_img = p_img.transpose(2, 0, 1) # 3 H W

        if len(modal_x.shape) == 2:
            p_modal_x = p_modal_x[np.newaxis, ...]
        else:
            p_modal_x = p_modal_x.transpose(2, 0, 1)
    
        return p_img, p_modal_x