import os.path as osp

import cv2
import numpy as np
import torch.nn as nn

from smoke.dataset.waymo import getKMatrix


class Scenario(nn.Module):
    def __init__(self, args: dict, global_args: dict):
        super().__init__()
        # 3*3 np.ndarray
        self.K = getKMatrix(calib_idx=args["waymo_index"],
                            calib_path=osp.join(global_args["project_path"], args["calib_dir"]))
        # HWC -> RGB -> 0~255 np.ndarray
        self.scenario: np.ndarray = self.load_scenario(scenario_idx=args["waymo_index"],
                                                       scenario_path=osp.join(global_args["project_path"],
                                                                              args["scenario_dir"]))
        # HW list
        self.scenario_size = self.scenario.shape[:2]
        self.visualization = {"scenario": self.scenario.copy()}

    def forward(self):
        return self.K, self.scenario, self.scenario_size

    @staticmethod
    def load_scenario(scenario_idx, scenario_path):
        """
        load scenario image as np.ndarray.\n
        :param scenario_idx: idx.
        :param scenario_path: path.
        :return: HWC -> RGB -> 0~255 np.ndarray.
        """
        img_path = osp.join(scenario_path, scenario_idx + ".png")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
