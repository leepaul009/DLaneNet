# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This script is used to process the auto lane dataset."""

import numpy as np
from scipy import interpolate
from .auto_lane_spline_interp import spline_interp
from .auto_lane_codec_utils import Point, Lane, get_lane_list, delete_repeat_y
from .auto_lane_codec_utils import delete_nearby_point, trans_to_lane_with_type
from .auto_lane_codec_utils import get_lane_loc_list, gettopk_idx


class PointLaneCodec(object):
    """This is the class of PointLaneCodec, which generate the groudtruth of every image pair.

    :param input_width: the width of input image
    :type input_width: float
    :param input_height: the height of input image
    :type input_height: float
    :param anchor_stride: the stride of every anchor
    :type anchor_stride: int
    :param points_per_line: the number of points in line
    :type points_per_line: int
    :param class_num: the class num of all lines
    :type class_num: int
    :param anchor_lane_num: how many lanes of every anchor
    :type anchor_lane_num: int
    """

    def __init__(self, input_width, input_height, anchor_stride,
                 points_per_line, class_num, anchor_lane_num=1):
        self.input_width = input_width # 512
        self.input_height = input_height # 256
        self.feature_width = int(input_width / anchor_stride) # 32 512/16
        self.feature_height = int(input_height / anchor_stride) # 16 256/16
        self.points_per_line = points_per_line # 80
        self.pt_nums_single_lane = 2 * points_per_line + 1 # 161
        self.points_per_anchor = points_per_line / self.feature_height # 5 80/16
        self.interval = float(input_height) / points_per_line # 3.2 256 / 80
        self.feature_size = self.feature_width * self.feature_height # 512 32x16 grid size
        self.class_num = class_num # 2
        self.img_center_x = input_width / 2
        self.step_w = anchor_stride
        self.step_h = anchor_stride
        self.anchor_lane_num = anchor_lane_num # 1

    def encode_lane(self, lane_object, org_width, org_height):
        """Encode lane to target type.

        :param lane_object: lane annot
        :type lane_object: mon dict (a dict of special format)
        :param org_width: image width
        :type org_width: int
        :param org_height: image height
        :type org_height: int
        :return: gt_type: [576, class_num]
                 gt_loc:  [576, 145]
        :rtype: nd.array
        """
        s_x = self.input_width * 1.0 / org_width # 512/1280
        s_y = self.input_height * 1.0 / org_height # 256/720
        gt_lanes_list = get_lane_list(lane_object, s_x, s_y) # get lines each from img bottom to top
        if len(gt_lanes_list) < 1: # if no line
            # background image
            gt_lane_offset = np.zeros(shape=(self.feature_size, self.points_per_line * 2 + 1), dtype=float)
            gt_lane_type = np.zeros(shape=(self.feature_size, self.class_num), dtype=float)
            gt_lane_type[:, 0] = 1
            gt_loc = gt_lane_offset.astype(np.float32)
            gt_type = gt_lane_type.astype(np.float32)
            ### 
            all_anchor_count = np.zeros(shape=(self.feature_height, self.feature_width))
            anchor_instance = np.zeros(shape=(1, self.feature_height, self.feature_width))
        else:
            lane_set = trans_to_lane_with_type(gt_lanes_list)
            # sort_lanes = order_lane_x_axis(lane_set, self.input_height)
            # lane_set=
            # lane_set = ensure_yshape_lines_order(sort_lanes, self.input_width, self.input_height)
            all_anchor_count = np.zeros(shape=(self.feature_height, self.feature_width)) # 16x32
            # anchor_instance = np.zeros(shape=(self.feature_height, self.feature_width))
            all_anchor_distance = list()
            all_anchor_loc = list()
            all_anchor_list = list()

            for lane in lane_set:
                cur_line = lane.lane
                new_lane = delete_repeat_y(cur_line) # return [{x=,y=}...], from im bottom to top
                if len(new_lane) < 2: # one point line
                    startpos = -1
                    endpos = -1
                    x_list = []
                    y_list = []
                else:
                    interp_lane = spline_interp(lane=new_lane, step_t=1) # [{x,y}...] from bottom to top
                    # x_pt_list, y_pt_list = trans_to_pt_list(interp_lane)
                    x_pt_list, y_pt_list = delete_nearby_point(interp_lane) # del pt close enough in y(<1px)
                    x_pt_list = x_pt_list[::-1] 
                    y_pt_list = y_pt_list[::-1] # reverse y: from top to bottom
                    startpos, endpos, x_list, y_list = \
                        self.uniform_sample_lane_y_axis(x_pt_list, y_pt_list)
                if startpos == -1 or endpos == -1:
                    continue
                anchor_list, anchor_distance_result, gt_loc_list = \
                    self.get_one_line_pass_anchors(startpos, endpos, x_list, y_list, all_anchor_count)

                all_anchor_distance.append(anchor_distance_result) #
                all_anchor_loc.append(gt_loc_list) # 
                all_anchor_list.append(anchor_list) # 

            # process gt offset value
            #if self.anchor_lane_num == 1:
            gt_type, gt_loc, anchor_instance = self.get_one_lane_gt_loc_type(all_anchor_distance,
                                                                all_anchor_loc, 
                                                                all_anchor_count)
            #elif self.anchor_lane_num == 2:
            #    gt_type, gt_loc = self.get_two_lane_gt_loc_type(all_anchor_distance,
            #                                                    all_anchor_loc, all_anchor_count)

        return gt_type, gt_loc, anchor_instance

    def decode_lane_orig(self, predict_type, predict_loc, cls_thresh):
        """Decode lane to normal type.

        :param predict_type: class result of groundtruth
        :type predict_type: nd.array whose shape is [576, class_num]
        :param predict_loc: regression result of groundtruth
        :type predict_loc: nd.array whose shape is [576, 145]=[576, 72+1+72]
        :return: lane set
        :rtype: dict
        """
        lane_set = list()
        grid_set = list()
        for h in range(self.feature_height):
            for w in range(self.feature_width):
                index = h * self.feature_width + w
                prob = predict_type[index][1]
                if prob < cls_thresh:
                    continue
                down_anchor_lane = predict_loc[index, :self.points_per_line]
                up_anchor_lane = predict_loc[index, self.points_per_line:]
                relative_end_pos = up_anchor_lane[0]
                anchor_y_pos = int((self.feature_height - 1 - h) * self.points_per_anchor)
                anchor_center_x = (1.0 * w + 0.5) * self.step_w
                anchor_center_y = (1.0 * h + 0.5) * self.step_h
                up_lane = np.array([])
                down_lane = np.array([])
                end_pos = anchor_y_pos
                start_pos = anchor_y_pos
                # up anchor
                for i in range(self.points_per_line):
                    if i >= relative_end_pos or anchor_y_pos + i >= self.points_per_line:
                        break
                    rela_x = up_anchor_lane[1 + i]
                    abs_x = anchor_center_x + rela_x
                    abs_y = self.input_height - 1 - (anchor_y_pos + i) * self.interval
                    p = Point(abs_x, abs_y)
                    up_lane = np.append(up_lane, p)
                    end_pos = anchor_y_pos + i + 1
                # down anchor
                for i in range(anchor_y_pos):
                    rela_x = down_anchor_lane[i]
                    abs_x = anchor_center_x + rela_x
                    abs_y = self.input_height - 1 - (anchor_y_pos - 1 - i) * self.interval
                    p = Point(abs_x, abs_y)
                    down_lane = np.append(p, down_lane)
                    start_pos = anchor_y_pos - 1 - i

                if up_lane.size + down_lane.size >= 2:
                    lane = np.append(down_lane, up_lane)
                    lane_predict = Lane(prob, start_pos, end_pos,
                                        anchor_center_x, anchor_center_y, 1, lane)
                    lane_set.append(lane_predict)
                    grid_set.append(index)

        return lane_set, grid_set


    def decode_lane(self, predict_type, predict_loc, cls_thresh):
        lane_set = list()
        grid_set = list()
        for h in range(self.feature_height):
            for w in range(self.feature_width):
                index = h * self.feature_width + w
                prob = predict_type[index][1]
                if prob < cls_thresh:
                    continue
                down_anchor_lane = predict_loc[index, :self.points_per_line]
                up_anchor_lane = predict_loc[index, self.points_per_line:]
                relative_end_pos = up_anchor_lane[0]
                anchor_y_pos = int((self.feature_height - 1 - h) * self.points_per_anchor)
                anchor_center_x = (1.0 * w + 0.5) * self.step_w
                anchor_center_y = (1.0 * h + 0.5) * self.step_h
                # up_lane = np.array([])
                # down_lane = np.array([])
                up_down_lane = list()
                end_pos = anchor_y_pos
                start_pos = anchor_y_pos
                
                ### up anchor
                for i in range(self.points_per_line):
                    if i >= relative_end_pos or anchor_y_pos + i >= self.points_per_line:
                        break
                    if i == 9: # get only 3 points lower
                        break
                    rela_x = up_anchor_lane[1 + i]
                    abs_x = anchor_center_x + rela_x
                    abs_y = self.input_height - 1 - (anchor_y_pos + i) * self.interval
                    # check if x/y are valid
                    # if abs_x > (self.input_width-1) or abs_y > (self.input_height-1) or abs_x < .0 or abs_y < .0:
                    if abs_x > (self.input_width-1) or abs_x < 0:
                        continue
                    # p = Point(abs_x, abs_y)
                    p = [abs_x, abs_y]
                    # up_lane = np.append(up_lane, p)
                    up_down_lane.append(p)
                    end_pos = anchor_y_pos + i + 1
                
                ### down anchor
                for i in range(anchor_y_pos):
                    if i == 3: # get only 3 points lower
                        break
                    rela_x = down_anchor_lane[i]
                    abs_x = anchor_center_x + rela_x
                    abs_y = self.input_height - 1 - (anchor_y_pos - 1 - i) * self.interval
                    # check if x/y are valid
                    # if abs_x > (self.input_width-1) or abs_y > (self.input_height-1) or abs_x < .0 or abs_y < .0:
                    if abs_x > (self.input_width-1) or abs_x < 0:
                        continue
                    # p = Point(abs_x, abs_y)
                    p = [abs_x, abs_y]
                    # down_lane = np.append(p, down_lane)
                    up_down_lane.append(p)
                    start_pos = anchor_y_pos - 1 - i

                # if up_lane.size + down_lane.size >= 2:
                if len(up_down_lane) >= 2:
                    # lane = np.append(down_lane, up_lane)
                    # lane_predict = Lane(prob, start_pos, end_pos, anchor_center_x, anchor_center_y, 1, lane)
                    # lane_set.append(lane_predict)
                    # lane_set.append( up_down_lane )
                    lane_set.append( {"line":up_down_lane, "prob":prob} )
                    grid_set.append(index)

        return lane_set, grid_set

    def get_one_lane_gt_loc_type(self, all_anchor_distance, all_anchor_loc, all_anchor_count):
        """Get the location and type of one lane.

        :param all_anchor_distance: all anchors with distance
        :type all_anchor_distance: list of tuple
        :param all_anchor_loc: all anchor with correspond lane regression struct. [num_lines, num_anchors, 161]
        :type all_anchor_loc: list
        :param all_anchor_count: the mask of weather anchor hit the lane or not.
        :type all_anchor_count: list
        :return gt_type: the type of groundtruth
        :rtype gt_type: nd.array
        :return gt_loc: the regression of groundtruth
        :rtype gt_loc: nd.array
        """
        gt_lane_offset = np.zeros(shape=(self.feature_size, self.pt_nums_single_lane), dtype=float) # [512, 161]
        gt_lane_type = np.zeros(shape=(self.feature_size, self.class_num), dtype=float)
        gt_lane_type[:, 0] = 1
        gt_instance = np.zeros(shape=(1, self.feature_height, self.feature_width))

        for h in range(self.feature_height):
            for w in range(self.feature_width):
                index = h * self.feature_width + w
                cnt = all_anchor_count[h][w] # indicates how many lanes are determined by this anchor
                # given grid(h,w), get corresp anchor_loc and anchor_dist, might exist multiple lines'
                # gt_lid_list: correspond line id(from 1) of this grid(h,w)
                gt_loc_list, gt_dist_list, gt_lid_list = \
                    get_lane_loc_list(all_anchor_distance, all_anchor_loc, h, w)

                if cnt == 0:  # back ground
                    gt_lane_type[index, 0] = 1
                elif cnt == 1:  # single
                    gt_lane_type[index, 0] = 0
                    gt_lane_type[index, 1] = 1
                    gt_lane_offset[index, :self.pt_nums_single_lane] = gt_loc_list[0]
                    gt_instance[0, h, w] = gt_lid_list[0]
                else:  # choose one
                    ### TBD design a best method to choose one line for this grid
                    gt_lane_type[index, 0] = 0
                    gt_lane_type[index, 1] = 1
                    # choose small distance
                    line_loc_num = len(gt_loc_list)
                    line_dist_num = len(gt_dist_list)
                    assert (line_dist_num == line_loc_num)
                    [top_idx] = gettopk_idx(gt_dist_list) # get line which has smallest distance [x, img_center]
                    gt_lane_offset[index, :self.pt_nums_single_lane] = gt_loc_list[top_idx]
                    gt_instance[0, h, w] = gt_lid_list[top_idx]

        gt_loc = gt_lane_offset.astype(np.float32)
        gt_type = gt_lane_type.astype(np.float32)
        gt_ins = gt_instance.astype(np.float32)

        return gt_type, gt_loc, gt_ins

    # y_pt_list from top to bottom
    # return:
    #   startpos: 0 
    #   endpos:   pos(0~79)
    #   x/y_list: "n(~79) points from image bottom -> top"
    def uniform_sample_lane_y_axis(self, x_pt_list, y_pt_list):
        """Ensure y from bottom of image."""
        if len(x_pt_list) < 2 or len(y_pt_list) < 2:
            return -1, -1, [], []
        max_y = y_pt_list[-1] # bottom pt
        if max_y < self.input_height - 1: # 255
            y1 = y_pt_list[-2]
            y2 = y_pt_list[-1]
            x1 = x_pt_list[-2]
            x2 = x_pt_list[-1]

            # add points from points(max_y) to img bottom
            while max_y < self.input_height - 1: # 255
                y_new = max_y + self.interval # 3.2(256/80)
                x_new = x1 + (x2 - x1) * (y_new - y1) / (y2 - y1)
                x_pt_list.append(x_new)
                y_pt_list.append(y_new)
                max_y = y_new

        x_list = np.array(x_pt_list)
        y_list = np.array(y_pt_list)  # y from small to big
        if y_list.max() - y_list.min() < 5:  # filter < 5 pixel lane (too short line will be ignored)
            return -1, -1, [], []
        if len(y_list) < 4:
            tck = interpolate.splrep(y_list, x_list, k=1, s=0)
        else:
            tck = interpolate.splrep(y_list, x_list, k=3, s=0)
        startpos = 0 # img bottom
        endpos = int((self.input_height - y_list[0]) / self.interval) # fariest point
        if endpos > self.points_per_line - 1: # 79
            endpos = self.points_per_line - 1
        if startpos >= endpos:
            return -1, -1, [], []

        y_list = []
        expand_pos = endpos
        for i in range(startpos, expand_pos + 1):
            y_list.append(self.input_height - 1 - i * self.interval) # get i/80 point's y in img-aixs
        xlist = interpolate.splev(y_list, tck, der=0) # img bottom -> top

        for i in range(len(xlist)):
            if xlist[i] == 0:
                xlist[i] += 0.01

        return startpos, endpos, xlist, y_list # xlist potentially has negative item

    # outputs:
    # anchor_list: anchors(grid h/w) of a lane
    # anchor_distance_result: list[h,w,dist], dist between corresp pt_x and img_center_x, that an anchor has corresp pt
    # Gt_loc_list: list[pt-x-offsets of a lane], that each anchor has a [list of x offset] as a lane stored within it
    # anchor_count: each item corresp to a grid, [the value of an item] indicates how many lane in this grid(==item)
    # inputs:
    #   xlist, y_list: <=80 points
    #   anchor_count(grid): np[16x32]
    def get_one_line_pass_anchors(self, startpos, endpos, xlist, y_list, anchor_count):
        """Get one line pass all anchors."""
        anchor_list = []
        anchor_distance_result = []
        Gt_loc_list = []

        for i in range(0, endpos - startpos + 1): # from 0 to end_pos(~79)
            # get grid index(h, w)
            h = self.feature_height - 1 - int((startpos + i) * self.interval / self.step_h) # step_h=16
            w = int(xlist[i] / self.step_w)  # IndexError: list index out of range
            if h < 0 or h > self.feature_height - 1 or w < 0 or w > self.feature_width - 1:
                continue
            if (h, w) in anchor_list: # only visit each grid once
                continue
            # get center point(x/y) of this anchor(grid)
            anchor_y = (1.0 * h + 0.5) * self.step_h
            center_x = (1.0 * w + 0.5) * self.step_w

            # ensure anchor on same side of lane
            curr_y = self.input_height - 1 - i * self.interval
            # curr point should be upper of center point of this anchor(grid)
            if curr_y <= anchor_y:
                continue

            anchor_list.append((h, w))
            # center_y: point located in the bottom middle of this grid
            center_y = y_list[int(self.points_per_line / self.feature_height) * (self.feature_height - 1 - h)]

            # get lane offset
            loss_line = [0] * (self.points_per_line * 2 + 1)
            length = endpos - startpos + 1
            # offset up cur anchor
            # loss_line right half: [mid pt -> top pt] [empty]
            up_index = 0
            for j in range(0, length):
                if y_list[startpos + j] <= center_y:
                    loss_line[self.points_per_line + 1 + up_index] = xlist[j] - center_x
                    up_index += 1
            loss_line[self.points_per_line] = up_index # num_top_pts
            # offset done cur anchor
            # loss_line left half: [mid pt -> bot pt] [empty]
            down_index = length - up_index - 1
            for j in range(0, endpos - startpos + 1): # from bot to top
                if y_list[startpos + j] > center_y:
                    if xlist[j] - center_x == 0:
                        loss_line[down_index] = 0.000001
                    else:
                        loss_line[down_index] = xlist[j] - center_x
                    down_index -= 1

            Gt_loc_list.append(loss_line)
            anchor_count[h][w] += 1
            distance = xlist[i] - self.img_center_x
            anchor_distance_result.append((h, w, distance))

        return anchor_list, anchor_distance_result, Gt_loc_list

