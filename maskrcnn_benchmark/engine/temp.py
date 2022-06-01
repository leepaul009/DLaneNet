# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os

import torch
from tqdm import tqdm

# from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size, get_rank
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
# from .bbox_aug import im_detect_bbox_aug

import torch
from torch import nn
from torch.nn import functional as F
from collections import ChainMap
import ujson
import numpy as np
import pandas as pd


from maskrcnn_benchmark.data.utils.auto_lane_codec_utils import nms_with_pos, order_lane_x_axis
from maskrcnn_benchmark.data.utils.auto_lane_codec_utils import convert_lane_to_dict
from maskrcnn_benchmark.data.utils.auto_lane_codec_utils import calc_err_dis_with_pos

import maskrcnn_benchmark.engine.post_proc_util as post_proc_util



def multidict_split(bundle_dict):
    """Split multi dict to retail dict.

    :param bundle_dict: a buddle of dict
    :type bundle_dict: a dict of list
    :return: retails of dict
    :rtype: list
    """
    retails_list = [dict(zip(bundle_dict, i)) for i in zip(*bundle_dict.values())]
    return retails_list

# def compute_on_dataset(model, data_loader, device, bbox_aug, timer=None):
def compute_on_dataset(cfg, model, data_loader, device, timer=None, dataset=None):
    model.eval()
    results_dict = {}

    if cfg.DEBUG:
        det_results_dict = {}
    
    cpu_device = torch.device("cpu")
    for inference_idx, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        with torch.no_grad():
            if timer:
                timer.tic()
            #if bbox_aug:
            #    output = im_detect_bbox_aug(model, images, device)
            #else:
            #    output = model(images.to(device))
            predicts, instances, loss = model(images.to(device), targets, backdoor=True)
            
            if cfg.DEBUG:
                output, det_output = post_process(cfg, image_ids, predicts, instances[-1], dataset)
            else:
                output = post_process(cfg, image_ids, predicts, instances[-1], dataset)

            if timer:
                if not device.type == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            # output = [o.to(cpu_device) for o in output]

        results_dict.update(
            {img_id.item(): result for img_id, result in zip(image_ids, output)}
        )
        if cfg.DEBUG:
            det_results_dict.update(
                {img_id.item(): result for img_id, result in zip(image_ids, det_output)}
            )
        if inference_idx == 5 and cfg.DEBUG:
            break

    if cfg.DEBUG:
        return results_dict, det_results_dict
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation.")

    # convert to a list
    # predictions = [predictions[i] for i in image_ids]
    return predictions

def temp_func(
    cfg,
    model,
    data_loader,
    dataset_name,
    device="cuda",
    expected_results=(),            # val: [] 
    expected_results_sigma_tol=4,   # val: 4
    output_folder=None,
    output_file=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    # predictions = compute_on_dataset(model, data_loader, device, bbox_aug, inference_timer)
    if cfg.DEBUG:
        predictions, det_predictions = compute_on_dataset(cfg, model, data_loader, device, inference_timer, dataset=dataset)
    else:
        predictions = compute_on_dataset(cfg, model, data_loader, device, inference_timer, dataset=dataset)
    # wait for all processes to complete before measuring the time
    synchronize()



    '''
    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return
    '''
    if output_folder:
        output_file_path = "inference_results" if output_file==None else output_file
        outputs = {
            'predictions': predictions,
            'test_images': dataset.test_imgs
        }
        output_file_path = "{}_r{}.pth".format(output_file_path, get_rank())
        # torch.save(outputs, os.path.join(output_folder, "results.pth"))
        torch.save(outputs, os.path.join(output_folder, output_file_path))
        logger.info("Successfully save results in {}".format( 
            os.path.join(output_folder, output_file_path) ))

        if cfg.DEBUG:
            output_file_path = "det_inference_results" if output_file==None else "det_"+output_file
            output_file_path = "{}_r{}.pth".format(output_file_path, get_rank())
            torch.save(det_predictions, os.path.join(output_folder, output_file_path))

