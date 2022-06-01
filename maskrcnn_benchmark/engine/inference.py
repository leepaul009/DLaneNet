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
        images, image_ids = batch
        with torch.no_grad():
            if timer:
                timer.tic()
            #if bbox_aug:
            #    output = im_detect_bbox_aug(model, images, device)
            #else:
            #    output = model(images.to(device))
            predicts, instances = model(images.to(device))
            
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


def post_process(cfg, image_ids, predicts, instances, dataset):
    batch_size = instances.size(0)
    feature_size = 4
    decode_lane = dataset.codec_obj.decode_lane

    # to cpu
    predict_loc = predicts['predict_loc'].detach().cpu().numpy()
    predict_conf = F.softmax(predicts['predict_cls'], -1).detach().cpu().numpy()
    instances = instances.permute((0, 2, 3, 1)).contiguous() # =>[N, grid_h, grid_w, 4]
    instances = instances.view(batch_size, -1, feature_size) # =>[N, grid_h*grid_w, 4]
    predict_ins = instances.detach().cpu().numpy()

    results = []
    if cfg.DEBUG:
        detailed_result = []

    for index, _ in enumerate( range(batch_size) ):

        lane_set, grid_set = decode_lane(
            predict_type=predict_conf[index],
            predict_loc=predict_loc[index], 
            cls_thresh=0.6)

        instance = predict_ins[index] # [grids=512, 4]
        lane_grid = generate_result(instance, grid_set, thresh=0.6)
        
        regourped_lines = list()
        for it in lane_grid:
            line = list()
            for i in it: # candidates for this line
                line.append(lane_set[i])
                #line.extend( lane_set[i] )
            regourped_lines.append(line)

        if cfg.DEBUG:
            detailed_result.append(regourped_lines)

        if cfg.DO_POST_PROC:
            ### post process
            final_lines = post_proc_util.post_proc(regourped_lines,
                rx=1280/512., ry=720/256., stride=16.) # stride: ignore pts outside
            results.append(final_lines)
        else:
            results.append(regourped_lines)      
    
    if cfg.DEBUG:
        return results, detailed_result

    return results


def generate_result(instance, grid_set, thresh=0.6):

    grid_y, grid_x = 16, 32

    # confidance = confidance.reshape(grid_y, grid_x)
    # mask = confidance > thresh

    # grid_location = np.zeros((grid_y, grid_x, 2))

    # grid = grid_location[mask]
    # offset = offsets[mask]
    # feature = instance[mask] # [valid_num, 4]
    feature = instance[grid_set] # [num_decode_lines, 4]
    lane_feature = []
    lane_grid = []
    # x = []
    # y = []
    # for i in range(len(grid)):
    for i in range( feature.shape[0] ):
        if (np.sum(feature[i]**2))>=0: # for i-th valid grid's feature

            if len(lane_feature) == 0:
                # create new lane
                lane_feature.append(feature[i])
                lane_grid.append([i]) # append feat id
                #x.append([point_x])
                #y.append([point_y])
            else:
                # flag = 0
                # index = 0
                min_feature_index = -1
                min_feature_dis = 10000

                for feature_idx, j in enumerate(lane_feature):
                    dis = np.linalg.norm((feature[i] - j)**2)
                    if min_feature_dis > dis:
                        min_feature_dis = dis # dist to this lane feat
                        min_feature_index = feature_idx # lane id

                if min_feature_dis <= 0.08:
                    # update feat of this lane
                    # lane_feature[min_feature_index] = (lane_feature[min_feature_index]*len(x[min_feature_index]) + feature[i])\
                    #                                   / (len(x[min_feature_index])+1)
                    temp_feat = lane_feature[min_feature_index]
                    n_grid = len(lane_grid[min_feature_index]) 
                    # lane feat + cur grid feat
                    lane_feature[min_feature_index] = (temp_feat * n_grid + feature[i]) / (n_grid + 1)
                    lane_grid[min_feature_index].append(i)
                    #x[min_feature_index].append(point_x)
                    #y[min_feature_index].append(point_y)
                elif len(lane_feature) < 12:
                    # create new lane
                    lane_feature.append(feature[i])
                    lane_grid.append([i])
                    #x.append([point_x])
                    #y.append([point_y])
    #
    return lane_grid

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

def inference(
    cfg,
    model,
    data_loader,
    dataset_name,
    # iou_types=("bbox",),
    # box_only=False,
    # bbox_aug=False,
    device="cuda",
    expected_results=(),            # val: [] 
    expected_results_sigma_tol=4,   # val: 4
    output_folder=None,
    output_file=None,
    validation=False,
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
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )


    ###
    if cfg.DO_POST_PROC:
        logger.info("Start to create and save csv file...")
        global_id = []
        img = []
        index = []
        img_width = []
        img_height = []
        lane_prob = []
        solid_prob = []
        points = []
        solid_type = []
        images_path = dataset.test_imgs
        for img_id, pred_lines in tqdm( predictions.items() ):
            if len(pred_lines) == 0:
                global_id.append( img_id )
                img.append( images_path[img_id].split('/')[-1] )
                index.append( 0 )
                img_width.append(1280)
                img_height.append(720)
                lane_prob.append( 1.0 )
                solid_prob.append( 1.0 )
                solid_type.append( 'solid' )
                points.append( [[1,1],[2,2]] )
            for lid, pred_line in enumerate( pred_lines ):
                global_id.append( img_id )
                img.append( images_path[img_id].split('/')[-1] )
                index.append( lid )
                img_width.append(1280)
                img_height.append(720)
                lane_prob.append( 1.0 )
                solid_prob.append( 1.0 )
                solid_type.append( 'solid' )
                points.append( pred_line )
        # create csv buf
        cont_list = {'global_id': global_id, 'img':img, 'index':index, 'img_width':img_width, 'img_height':img_height,
                    'prob':lane_prob,'solid_prob':solid_prob, 'solid_type':solid_type,'points':points}
        df = pd.DataFrame(cont_list)

        # save csv file
        csv_fpath = 'inference_{}_r_{}.csv'.format( output_file, get_rank() )
        csv_fpath = os.path.join(output_folder, csv_fpath)
        #os.path.join(exp_root, report_root, csv_fpath)
        df.to_csv(csv_fpath, index=False)
        logger.info('Save inference output as {}'.format(csv_fpath))

        # save state data
        inf_out_fpath = "inference_{}_r_{}.pkl".format( output_file, get_rank() )
        inf_out_fpath = os.path.join(output_folder, inf_out_fpath)
        torch.save(cont_list, inf_out_fpath)
        logging.info('Save output data as {}'.format(inf_out_fpath))
        ###

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

    '''
    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
    '''