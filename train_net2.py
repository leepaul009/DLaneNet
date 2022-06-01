import os
import argparse
import logging
# from collections import OrderedDict

import torch
# import torch.nn as nn

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader_for_loss_list
from maskrcnn_benchmark.solver import make_lr_scheduler, make_cosine_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.modeling.detector import build_lane_detection_model_for_loss_list

from maskrcnn_benchmark.engine.trainer2 import do_train

from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
# from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
# import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')


def train(cfg, local_rank, distributed, save_dir):
    # model = build_detection_model(cfg)
    model = build_lane_detection_model_for_loss_list(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    # scheduler = make_lr_scheduler(cfg, optimizer)
    scheduler = make_cosine_lr_scheduler(cfg, optimizer)

    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16" # false
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    arguments = {}
    arguments["iteration"] = 0

    # output_dir = cfg.OUTPUT_DIR
    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, save_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)
    
    data_loader = make_data_loader_for_loss_list(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    test_period = cfg.SOLVER.TEST_PERIOD
    if test_period > 0:
        data_loader_val = make_data_loader_for_loss_list(cfg, is_train=False, is_distributed=distributed, is_for_period=True)
    else:
        data_loader_val = None

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    do_train(
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
    )
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Lane Detection Training")
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
        "--skip_test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
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
    # args = parser.parse_args()
    args = parse_args()

    cuids = [ args.gpu_beg+i for i in range(args.gpu_cnt)]
    logging.info("visible devices: {}".format(cuids))
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, cuids))
    # num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    num_gpus = args.gpu_cnt
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = os.path.join(cfg.OUTPUT_DIR, args.workspace_name)
    if output_dir:
        mkdir(output_dir)

    # logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger = setup_logger("maskrcnn_benchmark", None, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    # logger.info("Collecting env info (might take some time)")
    # logger.info("\n" + collect_env_info())
    logger.info("Loaded configuration file {}".format(args.config_file))
    '''
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    # logger.info("Running with config:\n{}".format(cfg))
    '''

    output_config_path = os.path.join(output_dir, 'config.yml')
    # logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    model = train(cfg, args.local_rank, args.distributed, output_dir)

    # if not args.skip_test:
    #     run_test(cfg, model, args.distributed)


if __name__ == "__main__":
    main()