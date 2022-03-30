import torch
import torchvision.ops as tvo

from smoke.smoke import Smoke
from smoke.utils import AffineUtils


class Loss:
    target_map = {"CAR": 0, "CYCLIST": 1, "WALKER": 2, "": -1}

    def __init__(self, args: dict) -> None:
        self._device = args["device"]
        self._type = args["loss"]["type"]
        self._threshold = args["loss"]["threshold"]
        self._iou = args["loss"]["iou"]
        self._target = []
        targets = args["target"]
        if isinstance(targets, list):
            for target in targets:
                self._target.append(self.target_map.get(target))
        elif isinstance(targets, str):
            self._target.append(targets)

    def forward(self, box_pseudo_gt: dict, box3d_branch: torch.Tensor, K=None, scenario_size=None):
        res = None
        box3d_branch_score_filtered = self._filter_with_threshold(box3d_branch, self._threshold)
        box3d_branch_target_filtered = self._filter_with_target(box3d_branch_score_filtered, self._target)
        if box3d_branch_target_filtered.shape[0] == 0:
            return None
        if "class" == self._type:
            res = self._get_class_loss(box3d_branch_target_filtered)
        if "2d_iou" == self._type:
            assert self._iou >= 0
            box_2d = box3d_branch_target_filtered.detach().clone() \
                if box3d_branch_target_filtered.requires_grad \
                else box3d_branch_target_filtered.clone()
            box_2d = box_2d[:, 2:6]
            res = self._get_2d_gt_iou_loss(box3d_branch=box3d_branch_target_filtered,
                                           iou_threshold=self._iou,
                                           box_2d_gt=box_pseudo_gt["2d"],
                                           box_2d=box_2d)
        # Get 2D Box from 3D Box
        if "2d_iou_fake" == self._type and K is not None:
            assert self._iou >= 0
            assert scenario_size is not None
            K = Smoke.getIntrinsicMatrix(K=K,
                                         is_inverse=False,
                                         device="cpu")
            _, box_2d = self._decode_boxes(box3d_branch=box3d_branch_target_filtered,
                                           K=K,
                                           ori_img_size=scenario_size)
            res = self._get_2d_gt_iou_loss(box3d_branch=box3d_branch_target_filtered,
                                           iou_threshold=self._iou,
                                           box_2d_gt=box_pseudo_gt["2d"],
                                           box_2d=box_2d)
        return res

    @staticmethod
    def _get_class_loss(box3d_branch: torch.Tensor):
        """
        get sum of scores.\n
        :param box3d_branch: smoke output branch.
        :return: torch.Tensor.
        """
        if box3d_branch.shape[0] == 0:
            return None
        loss = box3d_branch[:, -1].sum() * -1
        return loss

    @staticmethod
    def _get_2d_gt_iou_loss(box3d_branch: torch.Tensor, iou_threshold: float, box_2d_gt, box_2d):
        if box3d_branch.shape[0] == 0:
            return None
        # Calculate Indexes
        with torch.no_grad():
            if isinstance(box_2d_gt, list):
                box_2d_gt = torch.tensor(box_2d_gt, device=torch.device("cpu"))
                if len(box_2d_gt.shape) == 1:
                    box_2d_gt = box_2d_gt.unsqueeze(0)
                box_2d = box_2d.cpu()
            iou = tvo.box_iou(box_2d_gt, box_2d).squeeze()
            index_select = torch.nonzero(iou >= iou_threshold)
        if index_select.shape[0] > 0:
            return torch.max(box3d_branch[index_select, -1]) * -1
        else:
            return None

    def _get_3d_gt_iou_loss(self):

        pass

    @staticmethod
    def _filter_with_threshold(box3d_branch: torch.Tensor, threshold: float):
        """filter box3d_branch with score threshold"""
        if threshold >= 0 and box3d_branch.shape[0] > 0:
            keep_idx = torch.nonzero(box3d_branch[:, -1] > threshold).squeeze()
            box3d_branch_ = box3d_branch.index_select(0, keep_idx)
            return box3d_branch_
        else:
            return box3d_branch

    @staticmethod
    def _filter_with_target(box3d_branch: torch.Tensor, targets: list):
        """filter box3d_branch with target"""
        keep_idx = None
        if len(targets) == 1 and targets[0] == -1:
            return box3d_branch
        for target in targets:
            # do not filter
            if target == -1:
                continue
            if keep_idx is None:
                keep_idx = torch.nonzero(box3d_branch[:, 0].int() == target).squeeze()
            else:
                keep_idx = torch.cat((keep_idx,
                                      torch.nonzero(box3d_branch[:, 0].int() == target).squeeze()),
                                     0)
        box3d_branch_ = box3d_branch.index_select(0, keep_idx)
        return box3d_branch_

    @staticmethod
    def _decode_boxes(box3d_branch: torch.Tensor, K: torch.Tensor, ori_img_size) -> (torch.Tensor, torch.Tensor):
        """decode 3D Box to get 2D Box"""
        if box3d_branch.shape[0] == 0:
            return None
        with torch.no_grad():
            if box3d_branch.requires_grad:
                box3d_branch_copy = box3d_branch.detach().clone().cpu()
            else:
                box3d_branch_copy = box3d_branch.clone().cpu()
            pred_alpha = box3d_branch_copy[:, 1]
            pred_dimensions = box3d_branch_copy[:, 6:9].roll(shifts=1, dims=1)
            pred_locations = box3d_branch_copy[:, 9:12]
            pred_rotation_y = AffineUtils.alpha2rotation_y_N(pred_alpha, pred_locations[:, 0], pred_locations[:, 2])
            # N*2*8 -> N*8*2
            box3d = AffineUtils.recovery_3d_box(pred_rotation_y, pred_dimensions,
                                                pred_locations, K, ori_img_size).permute(0, 2, 1)
            box3d_ = box3d.clone()
            # N*4 4->[x_min, y_min, x_max, y_max]
            box2d = torch.cat((box3d_.min(dim=1).values, box3d_.max(dim=1).values), dim=1)

        return box3d, box2d

    @DeprecationWarning
    def get_center_point_loss(self):
        pass
