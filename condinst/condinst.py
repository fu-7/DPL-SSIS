#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2019-2021 Megvii Inc. All rights reserved.

import math
from operator import is_
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn

from cvpods.layers import ShapeSpec, cat, generalized_batched_nms, get_norm
from cvpods.modeling.box_regression import Shift2BoxTransform
from cvpods.modeling.losses import dice_loss, iou_loss, sigmoid_focal_loss_jit
from cvpods.modeling.meta_arch.fcos import Scale
from cvpods.modeling.meta_arch.retinanet import permute_to_N_HWA_K
from cvpods.modeling.nn_utils.feature_utils import aligned_bilinear
from cvpods.structures import Boxes, ImageList, Instances, polygons_to_bitmask
from cvpods.utils import comm, log_first_n


def permute_all_to_N_HWA_K_and_concat(
        box_cls, box_delta, box_center, box_parmas, param_count, num_classes=80,
):
    box_cls_flattened = [permute_to_N_HWA_K(x, num_classes) for x in box_cls]
    box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
    box_center_flattened = [permute_to_N_HWA_K(x, 1) for x in box_center]
    box_parmas_flattened = [permute_to_N_HWA_K(x, param_count) for x in box_parmas]

    box_cls = cat(box_cls_flattened, dim=1).reshape(-1, num_classes)
    box_delta = cat(box_delta_flattened, dim=1).reshape(-1, 4)
    box_center = cat(box_center_flattened, dim=1).reshape(-1, 1)
    box_parmas = cat(box_parmas_flattened, dim=1).reshape(-1, param_count)
    return box_cls, box_delta, box_center, box_parmas


def compute_ious(pred, target):
    """
    Args:
        pred: Nx4 predicted bounding boxes
        target: Nx4 target bounding boxes
        Both are in the form of FCOS prediction (l, t, r, b)
    """
    pred_left = pred[..., 0]
    pred_top = pred[..., 1]
    pred_right = pred[..., 2]
    pred_bottom = pred[..., 3]

    target_left = target[..., 0]
    target_top = target[..., 1]
    target_right = target[..., 2]
    target_bottom = target[..., 3]

    target_aera = (target_left + target_right).clamp_(min=0) * \
                  (target_top + target_bottom).clamp_(min=0)
    pred_aera = (pred_left + pred_right).clamp_(min=0) * \
                (pred_top + pred_bottom).clamp_(min=0)

    w_intersect = (torch.min(pred_left, target_left) + \
                  torch.min(pred_right, target_right)).clamp_(min=0)
    h_intersect = (torch.min(pred_bottom, target_bottom) + \
                  torch.min(pred_top, target_top)).clamp_(min=0)

    g_w_intersect = (torch.max(pred_left, target_left) + \
                    torch.max(pred_right, target_right)).clamp_(min=0)
    g_h_intersect = (torch.max(pred_bottom, target_bottom) + \
                    torch.max(pred_top, target_top)).clamp_(min=0)
    ac_uion = (g_w_intersect * g_h_intersect).clamp_(min=0)

    area_intersect = (w_intersect * h_intersect).clamp_(min=0)
    area_union = target_aera + pred_aera - area_intersect
    eps = torch.finfo(torch.float32).eps

    ious = (area_intersect) / (area_union.clamp(min=eps))
    gious = ious - (ac_uion - area_union) / ac_uion.clamp(min=eps)

    return ious, gious
