import os.path as osp

import copy
import sys

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import transforms

from smoke.utils.obstacle import Obstacle


class Smoke(nn.Module):

    def __init__(self, args: dict, global_args: dict) -> None:
        super().__init__()
        self.device = torch.device(args["device"])
        self.transform_params = args["transform"]
        self.scenario_size = args["scenario_size"]
        self.input_size = args["shape"]["shape_input"][1:3]
        self.confidence_threshold = args["confidence_threshold"]
        self.visualization = {}

        assert torch.device("cuda:0") == self.device
        # init
        model_path = osp.join(global_args["project_path"], args["model_path"])
        self.model_path = model_path
        self.model = None
        # if osp.exists(model_path):
        #     self._model = torch.jit.load(model_path, map_location=self.device)
        # else:
        #     self._model = None
        # input params
        self.K = self.getIntrinsicMatrix(K=args["K"],
                                         is_inverse=False,
                                         device=self.device)
        self.K_Inverse = self.getIntrinsicMatrix(K=args["K"],
                                                 device=self.device)
        self.ratio = self.getInputRatio(scenario_size=args["scenario_size"],
                                        input_size=self.input_size,
                                        down_ration=args["down_ratio"],
                                        device=self.device)
        # output tensor
        self.box3d_branch = None
        self.feat_branch = None

    def forward(self, scenario, is_training=False) -> (torch.Tensor, torch.Tensor):
        # preprocessing ... transform
        scenario_input = self.smoke_transform(scenario)
        # reload smoke model for each epoch
        self.model = torch.jit.load(self.model_path, map_location=self.device)

        if self.model is not None:
            if not is_training:
                self.model.eval()
            self.box3d_branch, self.feat_branch = self.model.forward(scenario_input, (self.K_Inverse, self.ratio))

        # ======================================= Visualization =======================================
        with torch.no_grad():
            if self.box3d_branch.requires_grad:
                vis_box3d_branch = self.box3d_branch.detach().clone().cpu()
            else:
                vis_box3d_branch = self.box3d_branch.clone().cpu()
            obstacle_list = Obstacle.decode(box3d_branch_data=vis_box3d_branch,
                                            k=self.K,
                                            confidence_score=self.confidence_threshold,
                                            ori_img_size=self.scenario_size)
            self.visualization["detection"] = obstacle_list
        # =============================================================================================
        return self.box3d_branch, self.feat_branch

    def smoke_transform(self, scenario) -> torch.Tensor:
        scenario_input = None
        if isinstance(scenario, np.ndarray):
            scenario_input = torch.tensor(scenario, device=self.device).float()
        elif isinstance(scenario, torch.Tensor):
            scenario_input = scenario
        assert scenario_input is not None
        # CHW
        scenario_input = scenario_input.permute((2, 0, 1))
        transform = transforms.Compose([
            transforms.Resize(size=tuple(self.input_size)),
            transforms.Normalize(mean=tuple(self.transform_params["mean"]), std=tuple(self.transform_params["std"]))
        ])
        scenario_input_ = transform(scenario_input)
        # BCHW
        scenario_input_ = scenario_input_.unsqueeze(0).float()
        return scenario_input_

    @staticmethod
    def getIntrinsicMatrix(K: np.ndarray, is_inverse=True, device=torch.device("cpu")):
        k = K
        if len(K.shape) == 2:
            k = K[None, :, :]
        if is_inverse:
            return torch.tensor(np.linalg.inv(k), device=device)
        return torch.tensor(k, device=device)

    @staticmethod
    def getInputRatio(scenario_size, input_size, down_ration, device=torch.device("cpu")):
        return torch.tensor(np.array([[down_ration[1] * scenario_size[1] / input_size[1],
                                       down_ration[0] * scenario_size[0] / input_size[0]]], np.float32),
                            device=device)
