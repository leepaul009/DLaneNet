import argparse
import os
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.backbone import build_backbone
from maskrcnn_benchmark.modeling.ld_head import build_gfn, build_heads
from maskrcnn_benchmark.modeling.hourglass import build_hourglass
from maskrcnn_benchmark.modeling.detector import build_lane_detection_model
from maskrcnn_benchmark.data import make_data_loader, make_hard_data_loader
import random
from maskrcnn_benchmark.utils.comm import get_world_size, synchronize, get_rank

# python  -m torch.distributed.launch --nproc_per_node=2 test_01.py --gpu_beg 0 --gpu_cnt 2 

parser = argparse.ArgumentParser(description="")
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--gpu_beg", type=int, default=9)
parser.add_argument("--gpu_cnt", type=int, default=1)
parser.add_argument("--debug", type=int, default=0)
parser.add_argument(
        "--out_file",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
args = parser.parse_args()

distributed = args.gpu_cnt > 1
if distributed:
    #torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend="gloo", init_method="env://"
    )

cfg.merge_from_file('e2e_mask_rcnn_R_50_FPN_1x.yaml')
cfg.SOLVER.IMS_PER_BATCH = 32

sampling_list = [0, 12, 44, 32, 9, 7, 555, 800, 3200, 1200, 15, 3, 2, 3200, 801]
print(len(sampling_list), sampling_list)

hard_data_loader = make_hard_data_loader(
    cfg,
    is_train=True,
    is_distributed=(get_world_size()>1),
    sampling_list=sampling_list,
    num_iters = 10,
)

liter = iter(hard_data_loader)
for i in range(10):
    _, _, ids = next(liter)
    print(i, get_rank(), ids)
'''
start_iter = 0
for iteration, (images, targets, image_ids) in enumerate(hard_data_loader, start_iter):
    print(get_rank(), image_ids)
    if iteration == 5:
        break
'''