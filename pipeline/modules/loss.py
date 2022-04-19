import os

import torch
import torch.nn.functional as F
import torchvision.ops as tvo

from smoke.obstacle import Obstacle
from tools.vis_utils import draw_3d_boxes, draw_2d_boxes, plot_img


class Loss:
    target_map = {"CAR": 0, "CYCLIST": 1, "WALKER": 2, "": -1}

    def __init__(self, args: dict) -> None:
        self._device = args["device"]
        self._type = args["loss"]["type"]
        self._threshold = float(args["loss"]["threshold"])
        self._iou = float(args["loss"]["iou"])
        self._radius = float(args["loss"]["radius"])
        self._target = []
        targets = args["target"]
        if isinstance(targets, list):
            for target in targets:
                self._target.append(self.target_map.get(target))
        elif isinstance(targets, str):
            self._target.append(targets)

    def forward(self, box_pseudo_gt: dict, box3d_branch: torch.Tensor, smoke=None, K=None, scenario_size=None):
        res = None
        box3d_branch_target_filtered = self._filter_with_target(box3d_branch, self._target)
        box3d_branch_score_filtered = self._filter_with_threshold(box3d_branch_target_filtered, self._threshold)
        box3d_branch_3d_radius_filtered = self._filter_with_3d_radius(box3d_branch_score_filtered,
                                                                      box_3d_gt=box_pseudo_gt["3d"],
                                                                      radius=self._radius)
        box3d_branch_filtered = self._filter_with_2d_iou(box3d_branch_3d_radius_filtered, box_pseudo_gt["2d"],
                                                         self._iou)

        # ========================================= Debug =========================================
        if bool(os.getenv("debug")) and smoke is not None:
            box3d_branch_debug = box3d_branch_filtered.detach().clone().cpu() if box3d_branch_filtered.requires_grad \
                else box3d_branch_filtered.clone().cpu()
            obstacle_list = Obstacle.decode(box3d_branch_data=box3d_branch_debug,
                                            k=smoke.visualization.get("K"),
                                            confidence_score=0,
                                            ori_img_size=smoke.visualization.get("scenario_size"))
            if len(obstacle_list) > 0:
                # 0~255 RGB HWC uint8
                detection_3d_plot = draw_3d_boxes(smoke.visualization.get("scenario"), obstacle_list)
                # 0~255 RGB HWC uint8
                detection_2d_plot = draw_2d_boxes(smoke.visualization.get("scenario"), obstacle_list)
                plot_img(detection_3d_plot, "detection_3d_-1")
                plot_img(detection_2d_plot, "detection_2d_-1")
        # =========================================================================================

        if box3d_branch_filtered is None or box3d_branch_filtered.shape[0] <= 0:
            return None
        if "score" == self._type:
            res = self._get_score_loss(box3d_branch_filtered)
        elif "2d_iou" == self._type:
            # TODO
            res = self._get_score_loss(box3d_branch_filtered)
        elif "3d" == self._type:
            res = self._get_3d_gt_loss(box3d_branch_filtered, box_3d_gt=box_pseudo_gt["3d"])
        elif "3d_score_mix" == self._type:
            res = self._get_score_loss(box3d_branch_filtered) + \
                  self._get_3d_gt_loss(box3d_branch_filtered, box_3d_gt=box_pseudo_gt["3d"])
        return res

    @staticmethod
    def _get_score_loss(box3d_branch: torch.Tensor):
        """
        get sum of scores.\n
        :param box3d_branch: smoke output branch.
        :return: torch.Tensor.
        """
        if box3d_branch.shape[0] == 0:
            return None
        loss = torch.max(box3d_branch[:, -1]) * -1
        return loss

    @staticmethod
    def _get_2d_gt_iou_loss(box3d_branch: torch.Tensor, iou_threshold: float, box_2d_gt):
        pass

    @staticmethod
    def _get_3d_gt_loss(box3d_branch: torch.Tensor, box_3d_gt):
        device = box3d_branch.device

        # prepare 3d gt information
        box_3d_gt_location = box_3d_gt.get('location')
        box_3d_gt_dimensions = box_3d_gt.get('dimensions')
        h_offset = box_3d_gt.get('h_offset')
        if h_offset is not None:
            box_3d_gt_location[1] += h_offset
        if isinstance(box_3d_gt_location, list):
            box_3d_gt_location = torch.tensor(box_3d_gt_location, device=device)
        else:
            box_3d_gt_location = box_3d_gt_location.to(device)
        if isinstance(box_3d_gt_dimensions, list):
            box_3d_gt_dimensions = torch.tensor(box_3d_gt_dimensions, device=device)
        else:
            box_3d_gt_dimensions = box_3d_gt_dimensions.to(device)
        # w, h, l -> h, l, w
        box_3d_gt_dimensions = torch.roll(box_3d_gt_dimensions, shifts=2, dims=0)

        # calculate weight
        score = box3d_branch[:, -1]
        weight = F.softmax(score, dim=0)

        dimensions_loss = F.l1_loss(input=box3d_branch[:, 6:9],
                                    target=box_3d_gt_dimensions.expand(box3d_branch.shape[0], 3),
                                    reduction='none')
        dimensions_loss = torch.sum(dimensions_loss, dim=1)
        location_loss = F.l1_loss(input=box3d_branch[:, 9:12],
                                  target=box_3d_gt_location.expand(box3d_branch.shape[0], 3),
                                  reduction='none')
        location_loss = torch.sum(location_loss, dim=1)
        loss = torch.mul(dimensions_loss + location_loss, weight.unsqueeze(1))
        loss = torch.sum(loss)
        return 1 / (loss + 1.0) * -1

    @staticmethod
    def _filter_with_threshold(box3d_branch: torch.Tensor, threshold: float):
        """filter box3d_branch with score threshold"""
        if box3d_branch is None:
            return None
        if threshold >= 0 and box3d_branch.shape[0] > 0:
            keep_idx = torch.flatten((torch.nonzero(box3d_branch[:, -1] > threshold)))
            if keep_idx.shape[0] <= 0:
                return None
            box3d_branch_ = box3d_branch.index_select(0, keep_idx)
            return box3d_branch_
        else:
            return box3d_branch

    @staticmethod
    def _filter_with_target(box3d_branch: torch.Tensor, targets: list):
        """filter box3d_branch with target"""
        if box3d_branch is None:
            return None
        keep_idx = None
        if len(targets) == 1 and targets[0] == -1:
            return box3d_branch
        for target in targets:
            # do not filter
            if target == -1:
                continue
            if keep_idx is None:
                keep_idx = torch.flatten(torch.nonzero(box3d_branch[:, 0].int() == target))
            else:
                keep_idx = torch.cat((keep_idx,
                                      torch.flatten(torch.nonzero(box3d_branch[:, 0].int() == target))),
                                     0)
        if keep_idx is None or keep_idx.shape[0] <= 0:
            return None
        box3d_branch_ = box3d_branch.index_select(0, keep_idx)
        return box3d_branch_

    @staticmethod
    def _filter_with_2d_iou(box3d_branch: torch.Tensor, box_2d_gt, iou_threshold: float):
        """filter box3d_branch with 2d iou threshold"""
        if box3d_branch is None:
            return None
        if iou_threshold < 0:
            return box3d_branch
        device = box3d_branch.device
        if isinstance(box_2d_gt, list):
            box_2d_gt = torch.tensor(box_2d_gt, device=device)
        else:
            box_2d_gt = box_2d_gt.to(device)
        if len(box_2d_gt.shape) == 1:
            box_2d_gt = box_2d_gt.unsqueeze(0)
        box_2d = box3d_branch[:, 2:6]
        iou = tvo.box_iou(box_2d_gt, box_2d).squeeze()
        if iou.ndim == 0:
            iou = iou.unsqueeze(0)
        keep_idx = torch.flatten(torch.nonzero(iou > iou_threshold))
        if keep_idx.shape[0] <= 0:
            return None
        box3d_branch_ = box3d_branch.index_select(0, keep_idx)
        return box3d_branch_

    @staticmethod
    def _filter_with_3d_radius(box3d_branch: torch.Tensor, box_3d_gt: dict, radius: float = 3.0):
        """filter box3d_branch with 3d location which is in a circle of radius"""
        if box3d_branch is None:
            return None
        if radius < 0:
            return box3d_branch
        device = box3d_branch.device
        box_3d_gt_location = box_3d_gt.get('location')
        h_offset = box_3d_gt.get('h_offset')
        if h_offset is not None:
            box_3d_gt_location[1] += h_offset
        if isinstance(box_3d_gt_location, list):
            box_3d_gt_location = torch.tensor(box_3d_gt_location, device=device)
        else:
            box_3d_gt_location = box_3d_gt_location.to(device)
        location = box3d_branch[:, 9:12]
        distances = F.pairwise_distance(location, box_3d_gt_location, p=2)
        keep_idx = torch.flatten(torch.nonzero(distances <= radius))
        if keep_idx.shape[0] <= 0:
            return None
        box3d_branch_ = box3d_branch.index_select(0, keep_idx)
        return box3d_branch_
