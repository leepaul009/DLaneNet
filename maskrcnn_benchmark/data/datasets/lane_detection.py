import os
import glob
import json
from PIL import Image


import numpy as np
import torch
import torchvision

from copy import deepcopy

# from maskrcnn_benchmark.structures.bounding_box import BoxList
# from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
# from .abstract import AbstractDataset

# from cityscapesscripts.helpers import csHelpers
from maskrcnn_benchmark.data.utils.auto_lane_utils import get_img_whc, imread, load_module
from maskrcnn_benchmark.data.utils.auto_lane_utils import load_lines, resize_by_wh, bgr2rgb, imagenet_normalize, load_json
from maskrcnn_benchmark.data.utils.auto_lane_pointlane_codec import PointLaneCodec 

from more_itertools import grouper
from imgaug.augmentables.lines import LineStringsOnImage
from imgaug.augmentables.lines import LineString as ia_LineString
import imgaug as ia
import imgaug.augmenters as iaa

#from maskrcnn_benchmark.data.datasets import hard_sampling
import random
#import logging

class LaneDetectionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_dir,
        ann_dir,
        split='train',
        #mode="mask",
        transforms=None,
        cfg=None,
        # min_area=0,
        # mini=None,
    ):
        """
        Arguments:
            img_dir: /path/to/leftImg8bit/      has to contain {train,val,test}
            ann_dir: /path/to/gtFine/           has to contain {train,val,test}
            split: "train" or "val" or "test"
            mode: "poly" or "mask", which annotation format to use
            transforms: apply transformations to input/annotation
            min_area: exclude intances below a specific area (bbox area)
            mini: limit the size of the dataset, so len(dataset) == mini for
                debugging purposes
        """
        assert split in ["train", "val", "test"]

        # img_dir = os.path.abspath(os.path.join(img_dir, split))
        # ann_dir = os.path.abspath(os.path.join(ann_dir, split))
        img_dir = os.path.abspath(os.path.join(img_dir))
        ann_dir = os.path.abspath(os.path.join(ann_dir))

        assert os.path.exists(img_dir), img_dir
        assert os.path.exists(ann_dir), ann_dr

        self.ann_dir = ann_dir
        self.img_dir = img_dir
        self.transforms = transforms

        # img_pattern = os.path.join(img_dir, "*", "*_leftImg8bit.png")
        # img_paths = sorted(glob.glob(img_pattern))

        # self.img_paths = img_paths
        # self.ann_paths = ann_paths
        self.len_dataset = 0
        self.ann = None
        self.max_lanes = -1e5
        self.max_points = -1e5
        
        if split != "test":
            with open(self.ann_dir, "r") as ann_file:
                self.ann = json.load(ann_file)
                # list of dict
                # # img_path
                # # lanes
                # boxes, segmentations, labels = self._processAnn(ann)
            for it in self.ann:
                self.max_lanes  = max( self.max_lanes, len(it['lanes']) )
                self.max_points = max( self.max_points, max( [ len(ln) for ln in it['lanes'] ] ) )
            self.len_dataset = len(self.ann)
        else:
            test_imgs = os.path.join(img_dir, '*.bmp')
            self.test_imgs = glob.glob(test_imgs)
            self.len_dataset = len(self.test_imgs)
            
        #####################################################################
        self.cfg = cfg
        self.im_h, self.im_w = cfg.INPUT.INPUT_IMAGE_SIZE #256, 512
        self.orig_img_hw = cfg.INPUT.ORIGINAL_IMAGE_SIZE #720, 1280
        anchor_stride = cfg.INPUT.ANNOTATION.ANCHOR_STRIDE # 16
        points_per_line = cfg.INPUT.ANNOTATION.POINTS_PER_LANE # 80
        class_num = cfg.INPUT.ANNOTATION.NUM_CLASS # 2

        self.grid_x = int(self.im_w / anchor_stride)
        self.grid_y = int(self.im_h / anchor_stride)
        self.grid_ratio_x = (1280 / self.im_w) * anchor_stride
        self.grid_ratio_y = (720 / self.im_h) * anchor_stride
        self.codec_obj = PointLaneCodec(input_width=self.im_w, 
                                        input_height=self.im_h,
                                        anchor_stride=anchor_stride, 
                                        points_per_line=points_per_line,#72,
                                        class_num=class_num)
        self.encode_lane = self.codec_obj.encode_lane
        self.decode_lane = self.codec_obj.decode_lane

        #self.hard_sampler = hard_sampling.hard_sampling()
        self.sampling_list = []

    def __len__(self):
        return self.len_dataset

    def get_img_info(self, index):
        # Reverse engineered from voc.py
        # All the images have the same size
        im_h, im_w = self.orig_img_hw
        img_path = self.ann[index]['img_path'] if self.ann else self.test_imgs[index]
        return {
            "height": im_h,
            "width": im_w,
            "idx": index,
            "img_path": self.img_dir + img_path + '.bmp',
            # "ann_path": self.ann_paths[index],
        }

    def __getitem__(self, idx):
        '''
        img_path = self.img_paths[idx]
        ann_path = self.ann_paths[idx]

        if self.mode == "mask":
            ann = torch.from_numpy(np.asarray(Image.open(ann_path)))
            # masks are represented with tensors
            boxes, segmentations, labels = self._processBinayMasks(ann)
        else:
            with open(ann_path, "r") as ann_file:
                ann = json.load(ann_file)
            # masks are represented with polygons
            boxes, segmentations, labels = self._processPolygons(ann)

        boxes, segmentations, labels = self._filterGT(boxes, segmentations, labels)

        if len(segmentations) == 0:
            empty_ann_path = self.get_img_info(idx)["ann_path"]
            print("EMPTY ENTRY:", empty_ann_path)
            # self.img_paths.pop(idx)
            # self.ann_paths.pop(idx)
            img, target, _ = self[(idx + 1) % len(self)]

            # just override this image with the next
            return img, target, idx

        img = Image.open(img_path)
        # Compose all into a BoxList instance
        target = BoxList(boxes, img.size, mode="xyxy")
        target.add_field("labels", torch.tensor(labels))
        masks = SegmentationMask(segmentations, img.size, mode=self.mode)
        target.add_field("masks", masks)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target, idx
        '''
        # print("xxxxxxxxxxxxxxxx {} {}".format( self.sampling_list, self.hard_sampler.get_num() ))
        # orig_idx = -1
        if self.ann:
            '''
            choose = random.random()
            #if (self.sampling_list != None and len(self.sampling_list) > self.cfg.SOLVER.IMS_PER_BATCH and choose < 0.15)\
            #    or (self.cfg.DEBUG and self.sampling_list != None and len(self.sampling_list)>0):
            if len(self.sampling_list)>0:
                # random.sample sampling_list
                # orig_idx = idx
                idx = random.sample(self.sampling_list, 1)[0]
                print("[DEBUG] sampling image idx: {}".format( idx ))
            '''

            ann_lanes = self.ann[idx]['lanes']
            self.ann_lanes = ann_lanes
            img_path = os.path.join(self.img_dir, self.ann[idx]['img_path'] + '.bmp')
            try:
                lane_object = self._to_curvelane_dict(ann_lanes)
                image_arr = imread(img_path)
                whc = get_img_whc(image_arr)
                image_arr, lane_object = _lane_argue(image=image_arr, lane_src=lane_object)
                encode_type, encode_loc, encode_ins = self.encode_lane(lane_object=lane_object,
                                                            org_width=whc['width'],
                                                            org_height=whc['height'])
            except Exception:
                lane_object = self._to_curvelane_dict(ann_lanes)
                image_arr = imread(img_path)
                whc = get_img_whc(image_arr)
                encode_type, encode_loc, encode_ins = self.encode_lane(lane_object=lane_object,
                                                            org_width=whc['width'],
                                                            org_height=whc['height'])
            network_input_image = bgr2rgb(resize_by_wh(img=image_arr, width=self.im_w, height=self.im_h))

            img = imagenet_normalize(img=network_input_image)
            img = np.transpose(img, (2, 0, 1)).astype('float32')
            gt_loc = encode_loc.astype('float32')
            gt_cls = encode_type.astype('float32')

            # gt_instance = self.make_ground_truth_instance(ann_lanes).astype('float32')
            gt_instance = self.make_ground_truth_instance2(encode_ins).astype('float32')

            img = torch.from_numpy(img).float() 
            gt_loc = torch.from_numpy(gt_loc).float()
            gt_cls = torch.from_numpy(gt_cls).float()
            gt_instance = torch.from_numpy(gt_instance).float()

            target = dict(gt_loc=gt_loc, gt_cls=gt_cls, gt_ins=gt_instance, encode_ins=encode_ins)

            return img, target, idx #, orig_idx
        else:
            # inference
            image_path = self.test_imgs[idx]
            image_arr = imread(image_path)
            network_input_image = bgr2rgb(resize_by_wh(img=image_arr, 
                                                        width=self.im_w, 
                                                        height=self.im_h))
            img = imagenet_normalize(img=network_input_image)
            img = np.transpose(img, (2, 0, 1)).astype('float32')
            img = torch.from_numpy(img).float() 
            return img, idx #, image_path

    def _to_curvelane_dict(self, culane_lines):
        curvelane_lines = []
        for culane_line_spec in culane_lines:
            # curvelane_lien_spec = [{'x': x, 'y': y} for x, y in grouper(map(float, culane_line_spec.split(' ')), 2)]
            curvelane_lien_spec = [{'x': x, 'y': y}  for x, y in culane_line_spec]
            curvelane_lines.append(curvelane_lien_spec)
        return dict(Lines=curvelane_lines)

    def _lane_argue(self, image, lane_src):
        lines_tuple = [[(float(pt['x']), float(pt['y'])) for pt in line_spec] for line_spec in lane_src['Lines']]
        lss = [ia_LineString(line_tuple_spec) for line_tuple_spec in lines_tuple]

        lsoi = LineStringsOnImage(lss, shape=image.shape)
        color_shift = iaa.OneOf([
            iaa.GaussianBlur(sigma=(0.5, 1.5)),
            iaa.LinearContrast((1.5, 1.5), per_channel=False),
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1 * 255), per_channel=0.5),
            iaa.WithColorspace(to_colorspace=iaa.CSPACE_HSV, from_colorspace=iaa.CSPACE_RGB,
                                children=iaa.WithChannels(0, iaa.Multiply((0.7, 1.3)))),
            iaa.WithColorspace(to_colorspace=iaa.CSPACE_HSV, from_colorspace=iaa.CSPACE_RGB,
                                children=iaa.WithChannels(1, iaa.Multiply((0.1, 2)))),
            iaa.WithColorspace(to_colorspace=iaa.CSPACE_HSV, from_colorspace=iaa.CSPACE_RGB,
                                children=iaa.WithChannels(2, iaa.Multiply((0.5, 1.5)))),
        ])
        posion_shift = iaa.SomeOf(4, [
            iaa.Fliplr(),
            iaa.Crop(percent=([0, 0.2], [0, 0.15], [0, 0], [0, 0.15]), keep_size=True),
            iaa.TranslateX(px=(-16, 16)),
            iaa.ShearX(shear=(-15, 15)),
            iaa.Rotate(rotate=(-15, 15))
        ])
        aug = iaa.Sequential([
            iaa.Sometimes(p=0.6, then_list=color_shift),
            iaa.Sometimes(p=0.6, then_list=posion_shift)
        ], random_order=True)
        batch = ia.Batch(images=[image], line_strings=[lsoi])
        batch_aug = list(aug.augment_batches([batch]))[0]  # augment_batches returns a generator
        image_aug = batch_aug.images_aug[0]
        lsoi_aug = batch_aug.line_strings_aug[0]
        lane_aug = [[dict(x=kpt.x, y=kpt.y) for kpt in shapely_line.to_keypoints()] for shapely_line in lsoi_aug]
        return image_aug, dict(Lines=lane_aug)

    # input: 
    #   encode_ins: [1, grid_h, grid_w]
    def make_ground_truth_instance2(self, ins):
        grid_size = self.grid_y * self.grid_x
        ground = np.zeros((1, grid_size, grid_size))
        for i in range(grid_size): #make gt
            ins = ins[ins > -1] # useless step since ins get 0 or bigger val
            gt_one = deepcopy(ins)
            if ins[i] > 0: # has line in cur grid
                gt_one[ins==ins[i]] = 1 # same instance
                gt_one[ins!=ins[i]] = 2 # different instance, same class
                gt_one[ins==0] = 3 # different instance, different class
                ground[0][i] += gt_one
        return ground

    def make_ground_truth_instance(self, lanes):# target_lanes, target_h):
        
        grid_y = self.grid_y
        grid_x = self.grid_x
        grid_size = grid_y * grid_x
        
        # 1280x720 => 512 256 => 32x16
        resize_ratio_x = self.grid_ratio_x
        resize_ratio_y = self.grid_ratio_y
        
        # ground = np.zeros((len(target_lanes), 1, grid_size, grid_size))
        ground = np.zeros((1, grid_size, grid_size))

        # for batch_index, batch in enumerate(target_lanes):
        temp = np.zeros((1, grid_y, grid_x))
        lane_cluster = 1
        for lane_index, lane in enumerate(lanes):
            previous_x_index = 0
            previous_y_index = 0
            for point_index, (pt_x, pt_y) in enumerate(lane):
                if pt_x > 0:
                    x_index = int(pt_x / resize_ratio_x)
                    # y_index = int(target_h[batch_index][lane_index][point_index]/self.p.resize_ratio)
                    y_index = int(pt_y / resize_ratio_y)
                    temp[0][y_index][x_index] = lane_cluster
                if previous_x_index != 0 or previous_y_index != 0: #interpolation make more dense data
                    temp_x = previous_x_index
                    temp_y = previous_y_index
                    while False:
                        delta_x = 0
                        delta_y = 0
                        temp[0][temp_y][temp_x] = lane_cluster
                        if temp_x < x_index:
                            temp[0][temp_y][temp_x+1] = lane_cluster
                            delta_x = 1
                        elif temp_x > x_index:
                            temp[0][temp_y][temp_x-1] = lane_cluster
                            delta_x = -1
                        if temp_y < y_index:
                            temp[0][temp_y+1][temp_x] = lane_cluster
                            delta_y = 1
                        elif temp_y > y_index:
                            temp[0][temp_y-1][temp_x] = lane_cluster
                            delta_y = -1
                        temp_x += delta_x
                        temp_y += delta_y
                        if temp_x == x_index and temp_y == y_index:
                            break
                if pt_x > 0:
                    previous_x_index = x_index
                    previous_y_index = y_index
            lane_cluster += 1

        for i in range(grid_size): #make gt
            temp = temp[temp > -1]
            gt_one = deepcopy(temp)
            if temp[i]>0:
                gt_one[temp==temp[i]] = 1   #same instance
                if temp[i] == 0:
                    gt_one[temp!=temp[i]] = 3 #different instance, different class
                else:
                    gt_one[temp!=temp[i]] = 2 #different instance, same class
                    gt_one[temp==0] = 3 #different instance, different class
                ground[0][i] += gt_one

        return ground
'''

    def _filterGT(self, boxes, segmentations, labels):
        filtered_boxes = []
        filtered_segmentations = []
        filtered_labels = []
        assert len(segmentations) == len(labels) == len(boxes)

        for box, segmentation, label in zip(boxes, segmentations, labels):
            xmin, ymin, xmax, ymax = box
            area = (xmax - xmin) * (ymax - ymin)
            if area < self.min_area:
                continue

            filtered_boxes.append(box)
            filtered_segmentations.append(segmentation)
            filtered_labels.append(label)

        if len(filtered_boxes) < 1:
            filtered_boxes = torch.empty(0, 4)

        return filtered_boxes, filtered_segmentations, filtered_labels

    def _processPolygons(self, ann):
        # For a single object polygon annotations are stored in CityScapes like
        # [[x1, y1], [x2, y2]...] and we need them in the following format:
        # [x1, y1, x2, y2, x3, y3 ...]
        polys = []
        labels = []
        boxes = []

        def poly_to_tight_box(poly):
            xmin = int(min(poly[::2]))
            ymin = int(min(poly[1::2]))
            xmax = int(max(poly[::2]))
            ymax = int(max(poly[1::2]))
            bbox = xmin, ymin, xmax, ymax
            return bbox

        for inst in ann["objects"]:
            label = inst["label"]
            if label not in self.CLASSES:
                continue

            label = self.name_to_id[label]

            cityscapes_poly = inst["polygon"]
            poly = []
            for xy in cityscapes_poly:
                # Equivalent with `poly += xy` but this is more verbose
                x = xy[0]
                y = xy[1]
                poly.append(x)
                poly.append(y)

            # In CityScapes instances are described with single polygons only
            box = poly_to_tight_box(poly)

            boxes.append(box)
            polys.append([poly])
            labels.append(label)

        if len(boxes) < 1:
            boxes = torch.empty(0, 4)

        return boxes, polys, labels

    def _processBinayMasks(self, ann):
        boxes = []
        masks = []
        labels = []

        def mask_to_tight_box(mask):
            a = mask.nonzero()
            bbox = [
                torch.min(a[:, 1]),
                torch.min(a[:, 0]),
                torch.max(a[:, 1]),
                torch.max(a[:, 0]),
            ]
            bbox = list(map(int, bbox))
            return bbox  # xmin, ymin, xmax, ymax

        # Sort for consistent order between instances as the polygon annotation
        instIds = torch.sort(torch.unique(ann))[0]
        for instId in instIds:
            if instId < 1000:  # group labels
                continue

            mask = ann == instId
            label = int(instId / 1000)
            label = self.cityscapesID_to_ind[label]
            box = mask_to_tight_box(mask)

            boxes.append(box)
            masks.append(mask)
            labels.append(label)

        return boxes, masks, labels

'''
