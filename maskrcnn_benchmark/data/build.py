# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import bisect
import copy
import logging

import torch.utils.data
from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.miscellaneous import save_labels

from . import datasets as D
from . import samplers

from .collate_batch import BatchCollator, BBoxAugCollator
from .transforms import build_transforms
import os

def build_dataset(cfg, dataset_list, transforms, dataset_catalog, is_train=True):
    """
    Arguments:
        dataset_list (list[str]): Contains the names of the datasets, i.e.,
            coco_2014_train, coco_2014_val, etc
        transforms (callable): transforms to apply to each (image, target) sample
        dataset_catalog (DatasetCatalog): contains the information on how to
            construct a dataset.
        is_train (bool): whether to setup the dataset for training or testing
    """
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )
    datasets = []
    for dataset_name in dataset_list:
        data = dataset_catalog.get(dataset_name)
        factory = getattr(D, data["factory"])
        args = data["args"]
        # for COCODataset, we want to remove images without annotations
        # during training
        if data["factory"] == "COCODataset":
            args["remove_images_without_annotations"] = is_train
        if data["factory"] == "PascalVOCDataset":
            args["use_difficult"] = not is_train
        args["transforms"] = transforms
        args["cfg"] = cfg
        # make dataset from factory
        dataset = factory(**args)
        datasets.append(dataset)

    # for testing, return a list of datasets
    if not is_train:
        return datasets

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = D.ConcatDataset(datasets)

    return [dataset]


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_weighted_sampler(dataset, shuffle, distributed, input_path):
    if not os.path.exists(input_path):
        print("ERROR: {} not exists.".format(input_path))
    img_loss_list = torch.load(input_path, map_location=torch.device("cpu")) 

    weights = [0.2, 0.8, 0.02, 0.]
    weights = torch.tensor(weights, dtype=torch.float)
    classes_for_imgs = []
    account = [0, 0, 0, 0]
    for idx, loss in img_loss_list: # List[(id, loss), ...]
        if loss <= 3.:
            classes_for_imgs.append(0)
            account[0] += 1
        elif loss > 3. and loss <= 7.:
            classes_for_imgs.append(1)
            account[1] += 1
        elif loss > 7. and loss <= 10.:
            classes_for_imgs.append(2)
            account[2] += 1
        else:
            classes_for_imgs.append(3)
            account[3] += 1
    samples_weights = weights[classes_for_imgs]

    print("loss(<=3) has {}, loss(3~7) has {}, other has {}".format( account[0], account[1], account[2]+account[3] ))

    if distributed:
        return samplers.DistributedWeightedSampler(dataset, samples_weights, shuffle=shuffle)
    else:
        # return torch.utils.data.sampler.RandomSampler(dataset)
        #return torch.utils.data.WeightedRandomSampler(weights=samples_weights, 
        #                               num_samples=len(samples_weights), 
        #                               replacement=True)
        return samplers.DistributedWeightedSampler(dataset, samples_weights, shuffle=shuffle)
        

def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized

def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        img_info = dataset.get_img_info(i)
        aspect_ratio = float(img_info["height"]) / float(img_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def make_batch_data_sampler(
    dataset, sampler, aspect_grouping, images_per_batch, 
    num_iters=None, # max_iter 
    start_iter=0):
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, images_per_batch, drop_uneven=False
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_batch, drop_last=False
        )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )   
    return batch_sampler


def make_data_loader(
    cfg, 
    is_train=True, 
    is_distributed=False, 
    start_iter=0, 
    is_for_period=False, 
    is_inference=False):

    num_gpus = get_world_size()
    if is_train:
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH # ex.16
        assert (
            images_per_batch % num_gpus == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = True 
        num_iters = cfg.SOLVER.MAX_ITER # ex.40000
    else:
        if is_inference:
            #
            images_per_batch = cfg.TEST.IMS_PER_BATCH
            assert (
                images_per_batch % num_gpus == 0
            ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
                images_per_batch, num_gpus)
            images_per_gpu = images_per_batch // num_gpus
            shuffle = False
            num_iters = None
            start_iter = 0
        else:
            images_per_batch = cfg.TEST.IMS_PER_BATCH
            assert (
                images_per_batch % num_gpus == 0
            ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
                images_per_batch, num_gpus)
            images_per_gpu = images_per_batch // num_gpus
            shuffle = False if not is_distributed else True
            num_iters = None
            start_iter = 0

    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else [] # 1

    paths_catalog = import_file(
        "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
    if is_inference:
        dataset_list = cfg.DATASETS.INFERENCE 

    # If bbox aug is enabled in testing, simply set transforms to None and we will apply transforms later
    # TBD: transforms can not be used ==> update target
    transforms = None # if not is_train and cfg.TEST.BBOX_AUG.ENABLED else build_transforms(cfg, is_train)
    # TBD: update get_item
    datasets = build_dataset(cfg, dataset_list, transforms, DatasetCatalog, is_train or is_for_period)

    '''
    if is_train:
        # save category_id to label name mapping
        save_labels(datasets, cfg.OUTPUT_DIR)
    '''

    # each data_loader => one dataset
    data_loaders = []
    for dataset in datasets:

        if cfg.DATALOADER.WEIGHTED_SAMPLE:
            sampler = make_weighted_sampler(dataset, shuffle, is_distributed, cfg.DATALOADER.WEIGHTED_SAMPLE_INPUT)
        else:
            sampler = make_data_sampler(dataset, shuffle, is_distributed)
        batch_sampler = make_batch_data_sampler(
            dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter
        )
        # collator = BBoxAugCollator() if not is_train and cfg.TEST.BBOX_AUG.ENABLED else \
        #     BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY) # ex.0/32
        num_workers = cfg.DATALOADER.NUM_WORKERS # ex.4

        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            # collate_fn=collator,
        )
        data_loaders.append(data_loader)
    if is_train or is_for_period:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders


def make_hard_data_sampler(dataset, shuffle, distributed, sampling_list):
    # if distributed:
    return samplers.HardDistributedSampler(distributed=distributed, 
        shuffle=shuffle, 
        sampling_list=sampling_list)
    # sampler = torch.utils.data.sampler.RandomSampler(dataset)
    # return sampler

def make_hard_batch_data_sampler(sampler, images_per_batch, num_iters=None, start_iter=0):
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, images_per_batch, drop_last=False
    )
    #'''
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    #'''
    return batch_sampler

def make_hard_data_loader(
    cfg, 
    is_train=True, 
    is_distributed=False, 
    sampling_list=None,
    start_iter=0, 
    num_iters=500,
    is_for_period=False, 
    is_inference=False):

    num_gpus = get_world_size()

    images_per_batch = cfg.SOLVER.IMS_PER_BATCH # ex.16
    assert (
        images_per_batch % num_gpus == 0
    ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
        images_per_batch, num_gpus)
    images_per_gpu = images_per_batch // num_gpus
    shuffle = True
    # num_iters = cfg.SOLVER.MAX_ITER # ex.40000

    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = False
    paths_catalog = import_file(
        "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    dataset_list = cfg.DATASETS.TRAIN
    transforms = None
    datasets = build_dataset(cfg, dataset_list, transforms, DatasetCatalog, is_train or is_for_period)

    data_loaders = []
    for dataset in datasets:
        sampler = make_hard_data_sampler(dataset, shuffle, is_distributed, sampling_list)
        batch_sampler = make_hard_batch_data_sampler(
            sampler, images_per_gpu, num_iters, start_iter
        )
        # collator = BBoxAugCollator() if not is_train and cfg.TEST.BBOX_AUG.ENABLED else \
        #     BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY) # ex.0/32
        num_workers = cfg.DATALOADER.NUM_WORKERS # ex.4

        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            # collate_fn=collator,
        )
        data_loaders.append(data_loader)
    if is_train or is_for_period:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders

def make_data_loader_for_loss_list(
    cfg, 
    is_train=True, 
    is_distributed=False, 
    start_iter=0, 
    is_for_period=False, 
    is_inference=False):

    num_gpus = get_world_size()
    if is_train:
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH # ex.16
        assert (
            images_per_batch % num_gpus == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        ######################1029-11###########################
        #shuffle = True 
        #num_iters = cfg.SOLVER.MAX_ITER # ex.40000
        shuffle = False
        num_iters = None
        ######################1029-11###########################
    else:
        if is_inference:
            #
            images_per_batch = cfg.TEST.IMS_PER_BATCH
            assert (
                images_per_batch % num_gpus == 0
            ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
                images_per_batch, num_gpus)
            images_per_gpu = images_per_batch // num_gpus
            shuffle = False
            num_iters = None
            start_iter = 0
        else:
            images_per_batch = cfg.TEST.IMS_PER_BATCH
            assert (
                images_per_batch % num_gpus == 0
            ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
                images_per_batch, num_gpus)
            images_per_gpu = images_per_batch // num_gpus
            shuffle = False if not is_distributed else True
            num_iters = None
            start_iter = 0

    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else [] # 1

    paths_catalog = import_file(
        "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
    if is_inference:
        dataset_list = cfg.DATASETS.INFERENCE 

    # If bbox aug is enabled in testing, simply set transforms to None and we will apply transforms later
    # TBD: transforms can not be used ==> update target
    transforms = None # if not is_train and cfg.TEST.BBOX_AUG.ENABLED else build_transforms(cfg, is_train)
    # TBD: update get_item
    datasets = build_dataset(cfg, dataset_list, transforms, DatasetCatalog, is_train or is_for_period)

    '''
    if is_train:
        # save category_id to label name mapping
        save_labels(datasets, cfg.OUTPUT_DIR)
    '''

    # each data_loader => one dataset
    data_loaders = []
    for dataset in datasets:
        sampler = make_data_sampler(dataset, shuffle, is_distributed)
        batch_sampler = make_batch_data_sampler(
            dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter
        )
        # collator = BBoxAugCollator() if not is_train and cfg.TEST.BBOX_AUG.ENABLED else \
        #     BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY) # ex.0/32
        num_workers = cfg.DATALOADER.NUM_WORKERS # ex.4

        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            # collate_fn=collator,
        )
        data_loaders.append(data_loader)
    if is_train or is_for_period:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders