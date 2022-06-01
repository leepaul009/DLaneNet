# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import remove_small_boxes

from ..utils import cat
from .utils import permute_and_flatten

class RPNPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RPN boxes, before feeding the
    proposals to the heads
    """

    def __init__(
        self,
        pre_nms_top_n,
        post_nms_top_n,
        nms_thresh,
        min_size,
        box_coder=None,
        fpn_post_nms_top_n=None,
        fpn_post_nms_per_batch=True,
    ):
        """
        Arguments:
            pre_nms_top_n (int)
            post_nms_top_n (int)
            nms_thresh (float)
            min_size (int)
            box_coder (BoxCoder)
            fpn_post_nms_top_n (int)
        """
        super(RPNPostProcessor, self).__init__()
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = min_size

        if box_coder is None:
            box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.box_coder = box_coder

        if fpn_post_nms_top_n is None:
            fpn_post_nms_top_n = post_nms_top_n
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.fpn_post_nms_per_batch = fpn_post_nms_per_batch

    def add_gt_proposals(self, proposals, targets):
        """
        Arguments:
            proposals: list[BoxList] (assume listSize=N)
            targets: list[BoxList]
        """
        # Get the device we're operating on
        device = proposals[0].bbox.device

        gt_boxes = [target.copy_with_fields([]) for target in targets]

        # later cat of bbox requires all fields to be present for all bbox
        # so we need to add a dummy for objectness that's missing
        for gt_box in gt_boxes:
            gt_box.add_field("objectness", torch.ones(len(gt_box), device=device))

        proposals = [
            cat_boxlist((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes) # ?for each of N
        ]

        return proposals # list[BoxList] BoxList.shape=[5*H*W*3+?, 4]

    def forward_for_single_feature_map(self, anchors, objectness, box_regression):
        """
        Arguments:
            anchors: list[BoxList], (assume list number = batchSize N)
            objectness: tensor of size N, A, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        device = objectness.device
        N, A, H, W = objectness.shape

        # modify tensor shape [N, A*1, H, W] => [N, H*W*A, 1] => [N, H*W*A]
        objectness = permute_and_flatten(objectness, N, A, 1, H, W).view(N, -1)
        objectness = objectness.sigmoid()

        # modify tensor shape [N, A*41, H, W] => [N, H*W*A, 4]
        box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)

        num_anchors = A * H * W
        pre_nms_top_n = min(self.pre_nms_top_n, num_anchors)
        objectness, topk_idx = objectness.topk(pre_nms_top_n, dim=1, sorted=True) # [N, top_k_elems(H*W*A)]

        batch_idx = torch.arange(N, device=device)[:, None]
        box_regression = box_regression[batch_idx, topk_idx] # [N, top_k_elems(H*W*A), 4]

        image_shapes = [box.size for box in anchors] # list(tuple)
        concat_anchors = torch.cat([a.bbox for a in anchors], dim=0) # list(3HW,4),list size=N => (N*AHW,4)
        concat_anchors = concat_anchors.reshape(N, -1, 4)[batch_idx, topk_idx]

        # box offsets + orig boxes => proposals(N*3HW, 4)
        proposals = self.box_coder.decode(
            box_regression.view(-1, 4), concat_anchors.view(-1, 4)
        )

        proposals = proposals.view(N, -1, 4) # => (N,3HW,4)

        result = []
        # for each img if a batch(N), image_shapes => input image size
        for proposal, score, im_shape in zip(proposals, objectness, image_shapes):
            boxlist = BoxList(proposal, im_shape, mode="xyxy")
            boxlist.add_field("objectness", score)
            # clip proposals to image_shapes
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            boxlist = boxlist_nms(
                boxlist,
                self.nms_thresh,
                max_proposals=self.post_nms_top_n,
                score_field="objectness",
            )
            result.append(boxlist)
        return result # N*BoxList

    def forward(self, anchors, objectness, box_regression, targets=None):
        """
        Arguments:
            anchors: list[list[BoxList]], (assume listSize=[N, 5])
            objectness: list[tensor], (assume listSize=[5])
            box_regression: list[tensor], (assume listSize=[5])
            targets (list[BoxList): ground-truth boxes present in the image (optional)
        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        sampled_boxes = []
        num_levels = len(objectness)
        anchors = list(zip(*anchors)) # list[N,5] => list[5,N]
        
        # for each pyramid layer
        # output sampled_boxes:list[list[BoxList]], shape=5*N*[BoxList]
        for a, o, b in zip(anchors, objectness, box_regression): 
            sampled_boxes.append(self.forward_for_single_feature_map(a, o, b))

        boxlists = list(zip(*sampled_boxes)) # convert to N*5*BoxList
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists] # convert to N*BoxList

        if num_levels > 1:
            boxlists = self.select_over_all_levels(boxlists)

        # append ground-truth bboxes to proposals
        if self.training and targets is not None:
            boxlists = self.add_gt_proposals(boxlists, targets)

        # list[BoxList], listSize=N, BoxList contains bbox which is Tensor[2*5*A*H*W, 4]
        return boxlists

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        # different behavior during training and during testing:
        # during training, post_nms_top_n is over *all* the proposals combined, while
        # during testing, it is over the proposals for each image
        # NOTE: it should be per image, and not per batch. However, to be consistent 
        # with Detectron, the default is per batch (see Issue #672)
        if self.training and self.fpn_post_nms_per_batch:
            objectness = torch.cat(
                [boxlist.get_field("objectness") for boxlist in boxlists], dim=0
            )
            box_sizes = [len(boxlist) for boxlist in boxlists]
            post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
            _, inds_sorted = torch.topk(objectness, post_nms_top_n, dim=0, sorted=True)
            inds_mask = torch.zeros_like(objectness, dtype=torch.bool)
            inds_mask[inds_sorted] = 1
            inds_mask = inds_mask.split(box_sizes)
            for i in range(num_images):
                boxlists[i] = boxlists[i][inds_mask[i]]
        else:
            for i in range(num_images):
                objectness = boxlists[i].get_field("objectness")
                post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
                _, inds_sorted = torch.topk(
                    objectness, post_nms_top_n, dim=0, sorted=True
                )
                boxlists[i] = boxlists[i][inds_sorted]
        return boxlists


def make_rpn_postprocessor(config, rpn_box_coder, is_train):
    # 2000, Number of top scoring RPN proposals to keep after combining proposals from all FPN levels
    fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 
    
    if not is_train:
        fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST

    # 12000, When FPN is used, this is *per FPN level* (not total)
    pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TRAIN
    # 2000, Number of top scoring RPN proposals to keep after applying NMS
    post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N_TRAIN
    
    if not is_train:
        pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TEST
        post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N_TEST
    
    # Apply the post NMS per batch (default) or per image during training
    # (default is True to be consistent with Detectron, see Issue #672)
    fpn_post_nms_per_batch = config.MODEL.RPN.FPN_POST_NMS_PER_BATCH
    nms_thresh = config.MODEL.RPN.NMS_THRESH
    min_size = config.MODEL.RPN.MIN_SIZE
    
    box_selector = RPNPostProcessor(
        pre_nms_top_n=pre_nms_top_n,
        post_nms_top_n=post_nms_top_n,
        nms_thresh=nms_thresh,
        min_size=min_size,
        box_coder=rpn_box_coder,
        fpn_post_nms_top_n=fpn_post_nms_top_n, # ex. 2000
        fpn_post_nms_per_batch=fpn_post_nms_per_batch, # ex. True
    )
    return box_selector
