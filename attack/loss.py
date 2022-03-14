import torch
import torch.nn as nn


class Loss(nn.Module):
    target_map = {"CAR": 0, "CYCLIST": 1, "WALKER": 2}

    def __init__(self, args: dict) -> None:
        super().__init__()
        self.device = args["device"]
        self.type = args["loss"]["type"]
        self.threshold = args["loss"]["threshold"]
        self.target = self.target_map[args["target"]]

    def forward(self, box3d_branch: torch.Tensor) -> torch.Tensor:
        if "class" == self.type:
            return self.get_class_loss(self.target, box3d_branch)
        if "class_threshold" == self.type:
            return self.get_class_loss_with_threshold(self.target, box3d_branch, self.threshold)
        return torch.tensor(0.0, device=self.device)

    @staticmethod
    def get_class_loss(target: int, box3d_branch: torch.Tensor):
        """
        get sum of scores given target class.\n
        :param target: classification class.
        :param box3d_branch: smoke output branch.
        :return: torch.Tensor.
        """
        index = torch.nonzero(box3d_branch[:, 0].int() == target).squeeze()
        loss_tensor = torch.index_select(box3d_branch[:, -1], 0, index)
        loss = loss_tensor.sum() * -1
        return loss

    @staticmethod
    def get_class_loss_with_threshold(target: int, box3d_branch: torch.Tensor, threshold: float):
        """a version of class_loss with score threshold"""
        keep_idx = torch.nonzero(box3d_branch[:, -1] > threshold).squeeze()
        box3d_branch_ = box3d_branch.index_select(0, keep_idx)
        return Loss.get_class_loss(target, box3d_branch_)

    def get_center_point_loss(self):
        pass

    def get_2d_gt_iou_loss(self):
        pass

    def get_3d_gt_iou_loss(self):
        pass
