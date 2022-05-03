import copy

import numpy as np
import torch
import torchvision.ops as tvo

from pipeline.modules.sample import Sample


class Metric(object):

    type_map = {0: "Car", 1: "Cyclist", 2: "Walker", -1: "None"}

    status_map = {0: "normal", 1: "class", 2: "score", 3: "location", -1: "init"}

    def __init__(self) -> None:
        # ground truth
        self._sample = None
        self._dimension_gt = None
        self._box_2d_gt = None
        # prediction
        self._class_type = -1
        self._score = -1
        self._location_pred = None
        self._dimension_pred = None
        self._box_2d_pred = None
        # metrics
        self._location_delta = None
        self._dimension_delta = None
        self._iou_2d = -1
        self._status = -1

    @property
    def sample(self):
        return self._sample

    @sample.setter
    def sample(self, sample_: Sample):
        self._sample = copy.deepcopy(sample_)

    @property
    def dimension_gt(self):
        return self._dimension_gt

    @dimension_gt.setter
    def dimension_gt(self, dimension_gt_: list):
        if isinstance(dimension_gt_, list) and len(dimension_gt_) == 3:
            self._dimension_gt = copy.deepcopy(dimension_gt_)
        else:
            raise ValueError("Wrong dimension_gt type.")

    @property
    def box_2d_gt(self):
        return self._box_2d_gt

    @box_2d_gt.setter
    def box_2d_gt(self, box_2d_gt_: list):
        if isinstance(box_2d_gt_, list) and len(box_2d_gt_) == 4:
            self._box_2d_gt = copy.deepcopy(box_2d_gt_)
        else:
            raise ValueError("Wrong box_2d_gt type.")

    @property
    def class_type(self):
        return self._class_type

    @class_type.setter
    def class_type(self, class_type_: int):
        if isinstance(class_type_, int) and class_type_ in self.type_map.keys():
            self._class_type = class_type_
        else:
            raise ValueError("Wrong class_type type.")

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, score_: float):
        if isinstance(score_, float) and 0 <= score_ <= 1:
            self._score = score_
        else:
            raise ValueError("Wrong score type.")

    @property
    def location_pred(self):
        return self._location_pred

    @location_pred.setter
    def location_pred(self, location_pred_: list):
        if isinstance(location_pred_, list) and len(location_pred_) == 3:
            self._location_pred = copy.deepcopy(location_pred_)
        else:
            raise ValueError("Wrong location_pred type.")

        # calculate metric
        np_location_gt = np.array(self._sample.location)
        np_location_pred = np.array(self._location_pred)
        np_location_delta = np_location_pred - np_location_gt
        self._location_delta = np_location_delta.tolist()

    @property
    def dimension_pred(self):
        return self._dimension_pred

    @dimension_pred.setter
    def dimension_pred(self, dimension_pred_: list):
        if isinstance(dimension_pred_, list) and len(dimension_pred_) == 3:
            self._dimension_pred = copy.deepcopy(dimension_pred_)
        else:
            raise ValueError("Wrong dimension_pred type.")

        # calculate metric
        np_dimension_gt = np.array(self._dimension_gt)
        np_dimension_pred = np.array(self._dimension_pred)
        np_dimension_delta = np_dimension_pred - np_dimension_gt
        self._dimension_delta = np_dimension_delta.tolist()

    @property
    def box_2d_pred(self):
        return self._box_2d_pred

    @box_2d_pred.setter
    def box_2d_pred(self, box_2d_pred_: list):
        if isinstance(box_2d_pred_, list) and len(box_2d_pred_) == 4:
            self._box_2d_pred = copy.deepcopy(box_2d_pred_)
        else:
            raise ValueError("Wrong box_2d_pred type.")

        # calculate metric
        tensor_box_2d_gt = torch.tensor(self._box_2d_gt)
        tensor_box_2d_pred = torch.tensor(self._box_2d_pred)
        tensor_iou_2d = tvo.box_iou(tensor_box_2d_gt[None, ...], tensor_box_2d_pred[None, ...])
        self._iou_2d = (torch.flatten(tensor_iou_2d)).item()

    @property
    def location_delta(self):
        return self._location_delta

    @property
    def dimension_delta(self):
        return self._dimension_delta

    @property
    def iou_2d(self):
        return self._iou_2d

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, status_):
        if status_ in self.status_map.keys():
            self._status = status_
        else:
            raise ValueError("Invalid status.")

