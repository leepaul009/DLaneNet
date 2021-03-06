# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import logging

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_lane_detection_model

from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
# from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

# Check if we can enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for mixed precision via apex.amp')

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Lane Detection Inference")
    parser.add_argument(
        "--config_file",
        default="e2e_mask_rcnn_R_50_FPN_1x.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--gpu_beg", type=int, default=9)
    parser.add_argument("--gpu_cnt", type=int, default=1)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    parser.add_argument(
        "--out_file",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--workspace_name", default="test", help="Experiment name")
    # parser.add_argument("--resume", action="store_true", help="Resume training")
    # parser.add_argument("--validation", action="store_true", help="Resume training")   
    return parser.parse_args()

def main():
    '''
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    '''
    args = parse_args()
    
    cuids = [ args.gpu_beg+i for i in range(args.gpu_cnt)]
    logging.info("visible devices: {}".format(cuids))
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, cuids))

    # num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    num_gpus = args.gpu_cnt
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # save_dir = ""
    save_dir = os.path.join(cfg.OUTPUT_DIR, args.workspace_name)
    if not os.path.exists(save_dir):
        mkdir(save_dir)
        print("Create dir:  {}".format(save_dir))
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    # logger.info(cfg)

    # logger.info("Collecting env info (might take some time)")
    # logger.info("\n" + collect_env_info())

    # model = build_detection_model(cfg)
    model = build_lane_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE) # GPU

    # Initialize mixed-precision if necessary
    use_mixed_precision = cfg.DTYPE == 'float16' # false
    amp_handle = amp.init(enabled=use_mixed_precision, verbose=cfg.AMP_VERBOSE)

    # output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=save_dir) #output_dir)
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    # if cfg.OUTPUT_DIR:
    if save_dir:
        for idx, dataset_name in enumerate(dataset_names):
            # output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            output_folder = os.path.join(save_dir, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    
    # data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed, is_inference=True)
    data_loaders_test = make_data_loader(cfg, is_train=False, is_distributed=distributed, is_inference=True)

    # for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
    for output_folder, dataset_name, data_loader_test in zip(output_folders, dataset_names, data_loaders_test):
        inference(
            cfg,
            model,
            data_loader_test,
            dataset_name=dataset_name,
            # iou_types=iou_types,
            # box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            # bbox_aug=cfg.TEST.BBOX_AUG.ENABLED,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
            output_file=args.out_file,
        )
        synchronize()


if __name__ == "__main__":
    main()
