import torch
import torch.nn.functional as F
from torch import nn


class GFNModule(nn.Module):
    def __init__(self, cfg, in_channels):
        super(GFNModule, self).__init__()

        out_channels = cfg.MODEL.GFN.INTERNAL_CHANNEL
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(3):
            l_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
            self.lateral_convs.append(l_conv)
        for i in range(2):
            fpn_conv = nn.Sequential(
                nn.Conv2d(out_channels*2, out_channels*2, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels*2),
                nn.ReLU(),
            )
            self.fpn_convs.append(fpn_conv)

        self.c2_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 1/2 Pool
        self.c4_maxpool = nn.MaxPool2d(kernel_size=5, stride=4, padding=1) # 1/4 Pool
        self.out_channels = out_channels * 8 # 512

    # features: pyramid features x5
    def forward(self, features):
        build_out = []

        i1, i2 = 1, 2
        laterals = [self.lateral_convs[i1]( features[i1] ), 
                    self.lateral_convs[i2]( features[i2] )]
        laterals[0] = self.c2_maxpool( laterals[0] )
        build_out.append( self.fpn_convs[0]( torch.cat((laterals[0], laterals[1]), 1) ) ) # 64*2=>128

        i1, i2 = 0, 2
        laterals = [self.lateral_convs[i1]( features[i1] ), 
                    self.lateral_convs[i2]( features[i2] )]
        laterals[0] = self.c4_maxpool( laterals[0] )
        build_out.append( self.fpn_convs[1]( torch.cat((laterals[0], laterals[1]), 1) ) )

        outs = torch.cat((features[2], torch.cat((build_out[0], build_out[1]), 1)), 1) # 256+128*2
        return outs # [N 512 18 32]


class LDHeadModule(nn.Module):
    def __init__(self, cfg, in_channels):
        super(LDHeadModule, self).__init__()

        in_channels = 512 # 
        base_channel = cfg.MODEL.GFH.INTERNAL_CHANNEL #64
        num_classes  = cfg.INPUT.ANNOTATION.NUM_CLASS #2
        lane_up_pts_num   = cfg.INPUT.ANNOTATION.POINTS_PER_LANE #80
        lane_down_pts_num = cfg.INPUT.ANNOTATION.POINTS_PER_LANE + 1 #81
        self.lane_up_pts_num = lane_up_pts_num
        self.lane_down_pts_num = lane_down_pts_num
        self.num_classes = num_classes

        BatchNorm = nn.BatchNorm2d

        self.upper_lane_pred = nn.Sequential(
            nn.Conv2d(in_channels, base_channel, kernel_size=1, bias=False),
            BatchNorm(base_channel),
            nn.ReLU(),
            nn.Conv2d(base_channel, lane_up_pts_num, kernel_size=1, stride=1)
        )

        self.lower_lane_pred = nn.Sequential(
            nn.Conv2d(in_channels, base_channel, kernel_size=1, bias=False),
            BatchNorm(base_channel),
            nn.ReLU(),
            nn.Conv2d(base_channel, lane_down_pts_num, kernel_size=1, stride=1) # 81
        )

        self.cls_logits = nn.Sequential(
            nn.Conv2d(in_channels, base_channel, kernel_size=1, bias=False),
            BatchNorm(base_channel),
            nn.ReLU(),
            nn.Conv2d(base_channel, num_classes, kernel_size=1, stride=1)
        )
        for index, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # 122 022
    def forward(self, input):
        
        # print("head input: {} ".format( input[0] ))
        # print("w: {} ".format( self.upper_lane_pred.state_dict() ))

        predict_up   = self.upper_lane_pred(input).permute((0, 2, 3, 1)) # 80
        predict_down = self.lower_lane_pred(input).permute((0, 2, 3, 1)) # 81
        predict_cls  = self.cls_logits(input).permute((0, 2, 3, 1)).contiguous()

        # print("head predict_up: {} ".format( predict_up[0] ))

        predict_loc = torch.cat([predict_down, predict_up], -1).contiguous()

        predict_loc = predict_loc.view(predict_loc.shape[0], -1, self.lane_up_pts_num + self.lane_down_pts_num)
        predict_cls = predict_cls.view(predict_cls.shape[0], -1, self.num_classes)

        result = dict(
            predict_cls=predict_cls,
            predict_loc=predict_loc
        )
        return result

# Lane Detection Head
def build_heads(cfg, in_channels):
    return LDHeadModule(cfg, in_channels)

# Grid Fusion Network
def build_gfn(cfg, in_channels): 
    return GFNModule(cfg, in_channels)
