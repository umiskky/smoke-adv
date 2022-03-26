import copy
import os
import os.path as osp

import cv2


class Scenario:
    def __init__(self, args: dict, scenario_indexes: list):
        self._project_path = os.getenv("project_path")
        # dict of HWC -> RGB -> 0~255 np.ndarray
        # HW list
        self._scenarios, self.scenario_size = self._load_scenarios(scenario_idxes=scenario_indexes,
                                                                   scenario_path=osp.join(self._project_path,
                                                                                          args["scenario_dir"]))
        # 0~255 RGB HWC uint8
        self.visualization = {"scenarios": copy.deepcopy(self._scenarios)}

    def forward(self, scenario_index):
        return self._scenarios[scenario_index], self._scenarios[scenario_index].shape[:2]

    @staticmethod
    def _load_scenarios(scenario_idxes, scenario_path) -> (dict, list):
        """
        load scenario image as np.ndarray.\n
        :param scenario_idxes: idxes.
        :param scenario_path: path.
        :return: dict of HWC -> RGB -> 0~255 np.ndarray.
        """
        scenarios = {}
        image_size = None
        for index in scenario_idxes:
            img_path = osp.join(scenario_path, index + ".png")
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            scenarios[index] = image
            if image_size is None:
                image_size = image.shape[:2]
        return scenarios, image_size
