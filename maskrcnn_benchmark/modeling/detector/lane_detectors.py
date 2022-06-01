from .generalized_detector import LaneDetector
from .generalized_detector_for_loss_list import LaneDetector_for_loss_list


_DETECTION_META_ARCHITECTURES = {"LaneDetector": LaneDetector}


def build_lane_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)


def build_lane_detection_model_for_loss_list(cfg):
    meta_arch = LaneDetector_for_loss_list
    return meta_arch(cfg)