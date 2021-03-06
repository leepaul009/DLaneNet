import torch
from torch import nn
from torch.nn import functional as F
# from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..ld_head import build_gfn, build_heads
from ..hourglass import build_hourglass

from ..line_detection_heads.curlane_head import build_grid_fusion_nets, build_lane_detect_heads
# from ..rpn.rpn import build_rpn
# from ..roi_heads.roi_heads import build_roi_heads
import numpy as np
# from maskrcnn_benchmark.data.datasets import hard_sampling

def find_k_th_small_in_a_tensor(target_tensor, k_th):
    """Like name, this function will return the k the of the tensor."""
    val, idxes = torch.topk(target_tensor, k=k_th, largest=False)
    return val[-1]

def huber_fun(x):
    """Implement of hunber function."""
    absx = torch.abs(x)
    r = torch.where(absx < 1, x * x / 2, absx - 0.5)
    return r

class LaneDetector(nn.Module):
    def __init__(self, cfg):
        super(LaneDetector, self).__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)

        if cfg.MODEL.HEAD_TYPE == 1:
            self.gfn     = build_grid_fusion_nets(cfg, self.backbone.out_channels)
            self.gf_head = build_lane_detect_heads(cfg, self.gfn.out_channels)
        else:
            self.gfn     = build_gfn(cfg, self.backbone.out_channels)
            self.gf_head = build_heads(cfg, self.gfn.out_channels)

        self.hg_head = build_hourglass(cfg, in_channels=self.backbone.out_channels)

        # self.rpn = build_rpn(cfg, self.backbone.out_channels)
        # self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        self.POINTS_PER_LANE      = cfg.INPUT.ANNOTATION.POINTS_PER_LANE # 80
        self.LANE_POINTS_NUM_DOWN = self.POINTS_PER_LANE # 80
        self.LANE_POINTS_NUM_UP   = self.POINTS_PER_LANE + 1 # 81
        self.LANE_POINT_NUM_GT    = self.POINTS_PER_LANE * 2 + 1 #161
        self.OFFSET_WEIGHT = 5
        self.LOGIT_WEIGHT = 10
        self.NEGATIVE_RATIO = 15

        # self.hard_sampler = None

    def forward(self, images, targets=None, im_ids=None):
        
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if self.training:
            #print("train.................")
            return self._forward_train(images, targets, im_ids)
        elif targets is not None:
            #print("val..................")
            return self._forward_train(images, targets, im_ids)
        elif targets is None: # do inference
            return self._forward_test(images)
        else:
            print("detection forward error!!!!!!!!!!!!!!!!!")
            raise ValueError("Detection forward error!")

    def _forward_train(self, images, targets, im_ids):
        features = self.backbone(images)
        # print("features: {} ".format( features[-2][0] ))
        proposals = self.gfn(features)
        result = self.gf_head(proposals)
        instances = self.hg_head(features)
        return self.loss_evaluator(result, instances, targets, im_ids)

    def _forward_test(self, images):
        features  = self.backbone(images)
        proposals = self.gfn(features)
        result    = self.gf_head(proposals)
        instances = self.hg_head(features)
        return result, instances

    def loss_evaluator(self, result, instances, targets, im_ids):
    
        loc_targets = targets['gt_loc'] # [N, num_anchors, 145]
        cls_targets = targets['gt_cls'] # [N, num_anchors, 2]
        ins_targets = targets['gt_ins'] # [N, 1, grids, grids]

        batch_size = loc_targets.size(0)
        anchor_size = loc_targets.size(1)

        loc_preds = result['predict_loc']
        cls_preds = result['predict_cls']

        cls_targets = cls_targets[..., 1].view(-1) # [N x num_anchors], non-bg
        pmask = cls_targets > 0 # [N x num_anchors] each item indicates corresp anchor is lane or bg
        nmask = ~ pmask
        fpmask = pmask.float()
        fnmask = nmask.float()
        cls_preds = cls_preds.view(-1, cls_preds.shape[-1]) # [N x num_anchors, 2]
        loc_preds = loc_preds.view(-1, loc_preds.shape[-1]) # [N x num_anchors, 145]
        loc_targets = loc_targets.view(-1, loc_targets.shape[-1]) # [N x num_anchors, 145]
        total_postive_num = torch.sum(fpmask)
        total_negative_num = torch.sum(fnmask)  # Number of negative entries to select
        negative_num = torch.clamp(total_postive_num * self.NEGATIVE_RATIO, max=total_negative_num, min=1).int()
        positive_num = torch.clamp(total_postive_num, min=1).int()

        ######################################################################## DIY BEG
        '''
        total_postive_num_batch = torch.sum( fpmask.view(batch_size, anchor_size), -1 )
        total_postive_num_batch = torch.clamp(total_postive_num_batch, min=1).int()
        '''
        ######################################################################## DIY END

        # cls loss begin
        bg_fg_predict = F.log_softmax(cls_preds, dim=-1)
        fg_predict = bg_fg_predict[..., 1] # non-bg loss
        bg_predict = bg_fg_predict[..., 0] # bg loss
        # find k-th hard-to-pred bg-anchor
        max_hard_pred = find_k_th_small_in_a_tensor(bg_predict[nmask].detach(), negative_num)
        # mask only consider top-k bad bg-anchors => bg cls loss
        # choose k that balance pos&neg
        fnmask_ohem = ( bg_predict <= max_hard_pred ).float() * nmask.float() # [[N x num_anchors]
        # total_cross_pos = -torch.sum(self.LOGIT_WEIGHT * fg_predict * fpmask)
        # total_cross_neg = -torch.sum(self.LOGIT_WEIGHT * bg_predict * fnmask_ohem)
        total_cross_pos_anchor = self.LOGIT_WEIGHT * fg_predict * fpmask
        total_cross_pos = -torch.sum(total_cross_pos_anchor)
        total_cross_neg_anchor = self.LOGIT_WEIGHT * bg_predict * fnmask_ohem
        total_cross_neg = -torch.sum(total_cross_neg_anchor)
        # class loss end

        ######################################################################## DIY BEG
        '''
        total_cross_pos_batch = -torch.sum( ( total_cross_pos_anchor ).view(batch_size, anchor_size), -1 )
        total_cross_neg_batch = -torch.sum( ( total_cross_neg_anchor ).view(batch_size, anchor_size), -1 )
        total_cross_pos_batch = (total_cross_pos_batch / total_postive_num_batch).detach().cpu().numpy()
        total_cross_neg_batch = (total_cross_neg_batch / total_postive_num_batch).detach().cpu().numpy()
        '''
        ######################################################################## DIY END

        # regression loss begin

        ######################################################################## DIY BEG
        curve_beg=7 * 32 + 0 # grid in 
        curve_mid=7 * 32 + 31 ###############!!! OPT-01
        # curve_end=9 * 32 + 31
        curve_end=8 * 32 + 31
        mid_pt = self.LANE_POINTS_NUM_DOWN + 1 # 81
        curve_weighted_mask = torch.zeros_like(loc_targets) # [N x num_anchors, 145]
        
        ### for all grid:
        # middle line(5pts)
        curve_weighted_mask[:, mid_pt : mid_pt+5] = 2 # mid->up
        # upper line(10pts)
        curve_weighted_mask[:, mid_pt+5 : mid_pt+15] = 1 # mid->up
        # lower line(10pts)
        curve_weighted_mask[:, 0 : 10] = 1 # mid->bot
        
        ### for middle grid:
        for i in range(batch_size):
            mid_grid1 = curve_beg + i*anchor_size
            mid_gridX = curve_mid + 1 + i*anchor_size ############!!! OPT-01
            mid_grid2 = curve_end + 1 + i*anchor_size
            # middle line(5pts)
            curve_weighted_mask[mid_grid1 : mid_grid2, mid_pt : mid_pt+5] = 20 # mid->up
            # upper line(5pts)
            curve_weighted_mask[mid_grid1 : mid_grid2, mid_pt+5 : mid_pt+10] = 10 # 10 mid->up
            # lower line(5pts)
            curve_weighted_mask[mid_grid1 : mid_grid2, 0 : 5] = 10 # mid->bot

            ###############!!! OPT-01Beg
            curve_weighted_mask[mid_grid1 : mid_gridX, mid_pt : mid_pt+5] = 40 # mid->up
            curve_weighted_mask[mid_grid1 : mid_gridX, 0 : 5] = 20 # mid->bot
            ###############!!! OPT-01End
        
        curve_weighted_mask[..., self.LANE_POINTS_NUM_DOWN] = 50 #50 #10 ###!!! OPT-01 50->55
        # print(" loc_targets: {} {} ".format(  batch_size, anchor_size ))
        ######################################################################## DIY END

        #length_weighted_mask = torch.ones_like(loc_targets) # def device to loc_targets
        #length_weighted_mask[..., self.LANE_POINTS_NUM_DOWN] = 50 #10
        valid_lines_mask = pmask.unsqueeze(-1).expand_as(loc_targets) # => [N x num_anchors, 1] => [N x num_anchors, 145]
        valid_points_mask = (loc_targets != 0) # ???
        # unified_mask = length_weighted_mask.float() * valid_lines_mask.float() * valid_points_mask.float()
        unified_mask = curve_weighted_mask.float() * valid_lines_mask.float() * valid_points_mask.float()
        smooth_huber = huber_fun(loc_preds - loc_targets) * unified_mask
        loc_smooth_l1_loss = torch.sum(smooth_huber, -1)
        point_num_per_gt_anchor = torch.sum(valid_points_mask.float(), -1).clamp(min=1)
        total_loc = torch.sum(loc_smooth_l1_loss / point_num_per_gt_anchor)
        # regression loss end
        
        

        ######################################################################## DIY BEG
        '''
        loc_loss_batch = torch.sum( ( loc_smooth_l1_loss / point_num_per_gt_anchor ).view(batch_size, anchor_size), -1 )
        loc_loss_batch = (loc_loss_batch / total_postive_num_batch).detach().cpu().numpy()
        '''
        ######################################################################## DIY END


        total_cross_pos = total_cross_pos / positive_num
        total_cross_neg = total_cross_neg / positive_num
        total_loc = total_loc / positive_num
       
        #compute loss for similarity #################
        
        feature_size = self.cfg.INPUT.ANNOTATION.FEATURE_SIZE # 4
        K1 = self.cfg.LOSS.INSTANCE.K1
        ALPHA = self.cfg.LOSS.INSTANCE.ALPHA
        BETA = self.cfg.LOSS.INSTANCE.BETA
        im_h, im_w = self.cfg.INPUT.INPUT_IMAGE_SIZE # 256, 512
        stride = self.cfg.INPUT.ANNOTATION.ANCHOR_STRIDE # 16
        grid_y = int(im_h / stride) # 16 
        grid_x = int(im_w / stride) # 32

        grid_size = grid_y * grid_x
        sisc_loss = .0
        disc_loss = .0
        
        #sisc_loss_batch = np.zeros((batch_size))
        #disc_loss_batch = np.zeros((batch_size))

        for instance in instances:
            feature_map = instance.view(batch_size, feature_size, 1, grid_size)
            feature_map = feature_map.expand(batch_size, feature_size, grid_size, grid_size)#.detach()

            point_feature = instance.view(batch_size, feature_size, grid_size, 1)
            point_feature = point_feature.expand(batch_size, feature_size, grid_size, grid_size)#.detach()

            distance_map = (feature_map-point_feature)**2 
            distance_map = torch.sum( distance_map, dim=1 ).view(batch_size, 1, grid_size, grid_size)

            # same instance (is same line)
            sisc_loss = sisc_loss + torch.sum(distance_map[ins_targets==1]) / torch.sum(ins_targets==1)
            
            # print("distance_map size: {} \n ins_targets size: {} ".format( distance_map.size(), ins_targets.size() ))
            ############################################################# calculate batch loss
            '''
            distance_map_batch = distance_map.view(batch_size,-1)
            ins_targets_batch = (ins_targets==1).view(batch_size,-1)
            sisc_loss_batch = sisc_loss_batch + \
                ( torch.sum( distance_map_batch[ins_targets_batch], -1 ) /
                  torch.sum( ins_targets_batch, -1 )
                ).detach().cpu().numpy()
            '''

            # different instance, same class (is different line)
            # count = (K1 - distance_map[ins_targets==2]) > 0
            # count = torch.sum(count).data
            disc_loss = disc_loss + \
                torch.sum((K1 - distance_map[ins_targets==2])[ (K1 - distance_map[ins_targets==2]) > 0 ])/\
                torch.sum(ins_targets==2)

            ############################################################# calculate batch loss
            '''
            ins_targets_batch = (ins_targets==2).view(batch_size,-1)
            disc_loss_batch = disc_loss_batch + \
                ( torch.sum( (K1 - distance_map_batch[ins_targets_batch])[(K1 - distance_map_batch[ins_targets_batch]) > 0], -1 )/
                  torch.sum( ins_targets_batch, -1 )
                ).detach().cpu().numpy()
            '''

        total_ins = ALPHA * sisc_loss + BETA * disc_loss

        ############################################################# calculate batch loss
        #total_ins_loss_batch = ALPHA * sisc_loss_batch + BETA * disc_loss_batch
        #overall_loss_batch = total_cross_pos_batch + total_cross_neg_batch + loc_loss_batch + total_ins_loss_batch


        
        return dict(
            loss_pos=total_cross_pos,
            loss_neg=total_cross_neg,
            loss_loc=total_loc*self.OFFSET_WEIGHT,
            loss_ins=total_ins,
        )

