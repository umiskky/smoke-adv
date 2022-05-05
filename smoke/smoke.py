import os
import os.path as osp

import numpy as np
import torch
from torchvision.transforms import transforms

from pipeline.modules.sample import Sample


class Smoke:

    def __init__(self, args: dict) -> None:
        self._device = args["device"]
        assert "cuda:0" == self._device
        self._transform_params = args["transform"]
        self._input_size = args["shape"]["shape_input"][1:3]
        self._down_ratio = args["down_ratio"]
        self._model_path = osp.join(os.getenv("project_path"), args["model_path"])

        self.visualization = {}

    def forward(self, scenario, sample: Sample, is_training=False) -> (torch.Tensor, torch.Tensor):
        """ data
        [scenario_idx, K, scale, rotation, translate, ambient_color, diffuse_color, specular_color, location]
        """
        # load smoke model for each step
        model = torch.jit.load(self._model_path, map_location=self._device)
        if not is_training:
            model.eval()

        # preprocessing ... transform
        scenario_input, scenario_size = self.smoke_transform(scenario)
        K_Inverse = self.getIntrinsicMatrix(K=sample.K_inverse, device=self._device)
        ratio = self.getInputRatio(scenario_size=scenario_size,
                                   input_size=self._input_size,
                                   down_ration=self._down_ratio,
                                   device=self._device)

        # forward
        box3d_branch, feat_branch = model.forward(scenario_input, (K_Inverse, ratio))

        # ======================================= Visualization =======================================
        with torch.no_grad():
            if box3d_branch.requires_grad:
                vis_box3d_branch = box3d_branch.detach().clone().cpu()
            else:
                vis_box3d_branch = box3d_branch.clone().cpu()
            if isinstance(scenario, torch.Tensor):
                if scenario.requires_grad:
                    vis_scenario = scenario.detach().clone().cpu().numpy()
                else:
                    vis_scenario = scenario.clone().cpu().numpy()
            else:
                vis_scenario = scenario
            self.visualization["detection"] = vis_box3d_branch
            self.visualization["K"] = self.getIntrinsicMatrix(K=sample.K, device="cpu")
            self.visualization["scenario_size"] = scenario_size
            # np.ndarray HWC
            self.visualization["scenario"] = vis_scenario
        # =============================================================================================
        return box3d_branch, feat_branch

    def smoke_transform(self, scenario) -> (torch.Tensor, list):
        scenario_input = None
        if isinstance(scenario, np.ndarray):
            if scenario.dtype == np.uint8:
                scenario_input = torch.tensor(scenario, device=self._device).float()
            else:
                # float 32 / float 64   0~1.0 -> 0~255.0
                scenario_input = torch.tensor((scenario * 255.0).astype(np.float32), device=self._device)
        elif isinstance(scenario, torch.Tensor):
            if scenario.dtype == torch.uint8:
                scenario_input = scenario.float()
            else:
                # float 32 / float 64   0~1.0 -> 0~255.0
                scenario_input = torch.mul(scenario, 255.0)
                scenario_input = scenario_input.float()
        assert scenario_input is not None
        scenario_size = scenario_input.shape[0:2]
        # CHW
        scenario_input = scenario_input.permute((2, 0, 1))
        transform = transforms.Compose([
            transforms.Resize(size=tuple(self._input_size)),
            transforms.Normalize(mean=tuple(self._transform_params["mean"]), std=tuple(self._transform_params["std"]))
        ])
        scenario_input_ = transform(scenario_input)
        # BCHW
        scenario_input_ = scenario_input_.unsqueeze(0).float()
        return scenario_input_, scenario_size

    @staticmethod
    def getIntrinsicMatrix(K: np.ndarray, device="cpu"):
        if len(K.shape) == 2:
            K = K[None, ...]
        return torch.tensor(K, device=device)

    @staticmethod
    def getInputRatio(scenario_size, input_size, down_ration, device="cpu"):
        return torch.tensor(np.array([[down_ration[1] * scenario_size[1] / input_size[1],
                                       down_ration[0] * scenario_size[0] / input_size[0]]], np.float32),
                            device=device)
