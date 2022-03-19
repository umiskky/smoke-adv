import torch
import torch.nn as nn
import torchvision.ops as tvo

from smoke.smoke import Smoke
from smoke.utils import affine_utils


class Loss(nn.Module):
    target_map = {"CAR": 0, "CYCLIST": 1, "WALKER": 2}

    def __init__(self, args: dict) -> None:
        super().__init__()
        self.device = args["device"]
        self.type = args["loss"]["type"]
        self.threshold = args["loss"]["threshold"]
        self.iou = args["loss"]["iou"]
        self.target = self.target_map[args["target"]]
        self.scenario_size = args["scenario_size"]
        self.K = Smoke.getIntrinsicMatrix(K=args["K"],
                                          is_inverse=False,
                                          device=torch.device("cpu"))

    def forward(self, box_pseudo_gt: dict, box3d_branch: torch.Tensor):
        box3d_branch_score_filtered = self.filter_with_threshold(box3d_branch, self.threshold)
        box3d_branch_target_filtered = self.filter_with_target(box3d_branch_score_filtered, self.target)
        if "class" == self.type:
            return self.get_class_loss(box3d_branch_target_filtered)
        if "2d_iou" == self.type:
            assert self.iou >= 0
            _, box_2d = self.decode_boxes(box3d_branch=box3d_branch_target_filtered,
                                          K=self.K,
                                          ori_img_size=self.scenario_size)
            return self.get_2d_gt_iou_loss(box3d_branch=box3d_branch_target_filtered,
                                           iou_threshold=self.iou,
                                           box_2d_gt=box_pseudo_gt["2d"],
                                           box_2d=box_2d)
        return torch.tensor(0.0, device=self.device)

    @staticmethod
    def get_class_loss(box3d_branch: torch.Tensor):
        """
        get sum of scores.\n
        :param box3d_branch: smoke output branch.
        :return: torch.Tensor.
        """
        loss = box3d_branch[:, -1].sum() * -1
        return loss

    @staticmethod
    def get_2d_gt_iou_loss(box3d_branch: torch.Tensor, iou_threshold: float, box_2d_gt, box_2d):
        # Calculate Indexes
        with torch.no_grad():
            if isinstance(box_2d_gt, list):
                box_2d_gt = torch.tensor(box_2d_gt, device=torch.device("cpu"))
                if len(box_2d_gt.shape) == 1:
                    box_2d_gt = box_2d_gt.unsqueeze(0)
                box_2d = box_2d.cpu()
            iou = tvo.box_iou(box_2d_gt, box_2d).squeeze()
            index_select = torch.nonzero(iou >= iou_threshold)
        return box3d_branch[index_select, -1].sum() * -1

    def get_3d_gt_iou_loss(self):
        pass

    @staticmethod
    def filter_with_threshold(box3d_branch: torch.Tensor, threshold: float):
        """filter box3d_branch with score threshold"""
        if threshold >= 0:
            keep_idx = torch.nonzero(box3d_branch[:, -1] > threshold).squeeze()
            box3d_branch_ = box3d_branch.index_select(0, keep_idx)
            return box3d_branch_
        else:
            return box3d_branch

    @staticmethod
    def filter_with_target(box3d_branch: torch.Tensor, target: int):
        """filter box3d_branch with target"""
        keep_idx = torch.nonzero(box3d_branch[:, 0].int() == target).squeeze()
        box3d_branch_ = box3d_branch.index_select(0, keep_idx)
        return box3d_branch_

    @staticmethod
    def decode_boxes(box3d_branch: torch.Tensor, K: torch.Tensor, ori_img_size) -> (torch.Tensor, torch.Tensor):
        with torch.no_grad():
            if box3d_branch.requires_grad:
                box3d_branch_copy = box3d_branch.detach().clone().cpu()
            else:
                box3d_branch_copy = box3d_branch.clone().cpu()
            pred_alpha = box3d_branch_copy[:, 1]
            pred_dimensions = box3d_branch_copy[:, 6:9].roll(shifts=1, dims=1)
            pred_locations = box3d_branch_copy[:, 9:12]
            pred_rotation_y = affine_utils.alpha2rotation_y_N(pred_alpha, pred_locations[:, 0], pred_locations[:, 2])
            # N*2*8 -> N*8*2
            box3d = affine_utils.recovery_3d_box(pred_rotation_y, pred_dimensions,
                                                 pred_locations, K, ori_img_size).permute(0, 2, 1)
            box3d_ = box3d.clone()
            # N*4 4->[x_min, y_min, x_max, y_max]
            box2d = torch.cat((box3d_.min(dim=1).values, box3d_.max(dim=1).values), dim=1)

        return box3d, box2d

    @DeprecationWarning
    def get_center_point_loss(self):
        pass
