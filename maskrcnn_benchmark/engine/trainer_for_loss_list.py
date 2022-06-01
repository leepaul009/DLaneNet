# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import os
import time

import torch
import torch.distributed as dist
from tqdm import tqdm

from maskrcnn_benchmark.data import make_data_loader, make_hard_data_loader
from maskrcnn_benchmark.utils.comm import get_world_size, synchronize, get_rank
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
# from maskrcnn_benchmark.engine.inference import inference # nms C
from maskrcnn_benchmark.data.datasets import hard_sampling
from apex import amp
import copy
import random

#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    cfg,
    model,
    data_loader,
    data_loader_val,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    test_period,
    arguments,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    # model.train()
    start_training_time = time.time()
    end = time.time()

    # corner_cases = []
    corner_case_sampler = hard_sampling.hard_sampler()
    # hard_sampler = hard_sampling.hard_sampling()
    HC_dataloader = None
    HC_dataloader_iter = None
    HC_UPDATE_PEROID = cfg.HARD_CASE_UPDATE_PERIOD # 800
    HC_AVAIL_NUM = 0
    min_sampling_num = cfg.SOLVER.IMS_PER_BATCH / get_world_size()
    
    model.eval()
    outputs = {}
    for iteration, (images, targets, image_ids) in enumerate( data_loader, start_iter ):
        iteration = iteration + 1
        arguments["iteration"] = iteration
        '''
        is_print_corner_case = False
        if random.random()<0.01:
            if HC_dataloader_iter is not None and\
            corner_case_sampler.get_num() >= min_sampling_num and\
            HC_AVAIL_NUM>0:
                images, targets, image_ids = next(HC_dataloader_iter)
                # print("[DEBUG] Iter {}, Get from hard case: rank {}, img_ids {}".format( iteration, get_rank(), image_ids.tolist() ))
                print("[DEBUG] Iter {} rank {} get from hard case for trainning".format( iteration, get_rank(), ))
                HC_AVAIL_NUM -= 1
                is_print_corner_case = True
        '''
        if cfg.DEBUG and iteration == (start_iter+10):
            print("step {}, start_iter {}".format(iteration, start_iter))
            print("type: {} {}".format( type(images), type(targets), ))
            print("size: {} {} {}".format( images.size(), 
                targets['gt_loc'].size(), targets['gt_cls'].size(), targets['gt_ins'].size(), ))
            checkpointer.save("model_debug", meters, **arguments)
            break
        
        data_time = time.time() - end

        images = images.to(device)
        targets = dict(gt_loc=targets['gt_loc'].to(device), 
                       gt_cls=targets['gt_cls'].to(device),
                       gt_ins=targets['gt_ins'].to(device) )
                       
        loss_dict, loss_batch = model(images, targets, image_ids)
        
        # update corner case
        for i, loss_per_batch in enumerate(loss_batch):
            '''
            node = hard_sampling.sampling_node(loss = loss_per_batch, 
                                               data = image_ids[i], 
                                               previous_node = None, 
                                               next_node = None)
            hard_sampler.insert(node)
            '''
            outputs[image_ids[i]] = loss_per_batch
            # corner_case_sampler.insert(img_id=image_ids[i], loss=loss_per_batch)
        # update sampling_list for each epoch,
        # and create a new dataloader(hardsampler) for next epoch
        '''
        if iteration % HC_UPDATE_PEROID == 0:
            HC_AVAIL_NUM = int(HC_UPDATE_PEROID*0.15)
            corner_cases = corner_case_sampler.get_ids()
            HC_dataloader = make_hard_data_loader(
                cfg,
                is_train=True,
                is_distributed=False, #(get_world_size()>1),
                sampling_list=corner_cases,
                num_iters = (HC_AVAIL_NUM),
            )
            HC_dataloader_iter = iter(HC_dataloader)
            print("[DEBUG] Iter {}, Rank {} create data loader from sampling_list with len {}"
                .format(iteration, get_rank(), len(corner_cases) ))
        '''   

        '''
        losses = sum(loss for loss in loss_dict.values())
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()
        # losses.backward()
        optimizer.step()
        scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        
        if iteration % 20 == 0 or iteration == max_iter or iteration == (start_iter+1):
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}/{max_iter}",
                        "{meters}",
                        "lr: {lr:.7f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    max_iter=max_iter,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), meters, **arguments)
        if iteration % 5000 == 0:
            torch.save(corner_case_sampler, "torner_case_{:07d}.pth".format(iteration) )
        '''

    torch.save(outputs, "ooooooooooooooooooo_{}.pth".format( get_rank() ) )