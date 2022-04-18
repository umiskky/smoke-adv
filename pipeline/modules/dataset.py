import codecs
import copy
import os
import os.path as osp
import random
from typing import List, Dict

import yaml

from pipeline.modules.sample import Sample
from smoke.waymo import getKMatrix


class Dataset:
    def __init__(self, args: dict) -> None:
        self._random_args = args["random"]
        self._project_path = os.getenv("project_path")
        _, self._logger_console = args["logger"]
        if self._project_path is None:
            if self._logger_console is not None:
                self._logger_console.error("The Project Path in env is None.")
            raise ValueError('The Project Path in env is None.')
        self._meta_file = osp.join(self._project_path, args["meta"])
        self._samples, self._scenario_indexes = self._load_meta_from_yaml(self._meta_file, osp.join(self._project_path, args["calib_dir"]))
        # Use for data
        self._counter = 0
        self._frequency = int(self._random_args.get("frequency"))
        self._train_data = None
        self._eval_data = None

    def __len__(self):
        return len(self._samples)

    @staticmethod
    def _load_meta_from_yaml(meta_file: str, calib_path: str):
        samples = []
        scenario_indexes = []
        with codecs.open(meta_file, 'r', 'utf-8') as f:
            dic: dict = yaml.load(f, Loader=yaml.FullLoader)
        if len(dic) > 0:
            for scenario_idx in dic.keys():
                scenario_indexes.append(scenario_idx)
                # only define scenario index
                if dic[scenario_idx] is None:
                    sample = Sample(scenario_idx)
                    sample.K = getKMatrix(calib_idx=scenario_idx,
                                          calib_path=calib_path)
                    samples.append(sample)
                    continue
                # only one location that model.matrix.translate is [int | float * 3],
                # the types of other parameters must be same as model.matrix.translate or None.
                model_matrix_translate = dic[scenario_idx]["model_matrix"]["translate"]
                if not isinstance(model_matrix_translate[0], list):
                    sample = Sample(scenario_idx)
                    sample.K = getKMatrix(calib_idx=scenario_idx,
                                          calib_path=calib_path)
                    sample.scale = dic[scenario_idx]["model_matrix"].get("scale")
                    sample.location = model_matrix_translate
                    sample.rotation = dic[scenario_idx]["model_matrix"].get("rotation")
                    sample.light_ambient_color = dic[scenario_idx]["light"].get("ambient_color")
                    sample.light_diffuse_color = dic[scenario_idx]["light"].get("diffuse_color")
                    sample.light_specular_color = dic[scenario_idx]["light"].get("specular_color")
                    sample.light_location = dic[scenario_idx]["light"].get("location")
                    samples.append(sample)
                    continue
                # multiple locations for one scenario that model.matrix.translate is [[int | float * 3] * n],
                # the types of other parameters must be same as model.matrix.translate or None or [int | float * 3].
                for index in range(len(model_matrix_translate)):
                    sample = Sample(scenario_idx)
                    sample.K = getKMatrix(calib_idx=scenario_idx,
                                          calib_path=calib_path)
                    # scale
                    scale_ = dic[scenario_idx]["model_matrix"].get("scale")
                    if isinstance(scale_, list):
                        sample.scale = scale_[index]
                    else:
                        sample.scale = scale_
                    # location
                    sample.location = model_matrix_translate[index]
                    # rotation
                    rotation_ = dic[scenario_idx]["model_matrix"].get("rotation")
                    if isinstance(rotation_[0], list):
                        sample.rotation = rotation_[index]
                    else:
                        sample.rotation = rotation_
                    # light_ambient_color
                    light_ambient_color_ = dic[scenario_idx]["light"].get("ambient_color")
                    if isinstance(light_ambient_color_[0], list):
                        sample.light_ambient_color = light_ambient_color_[index]
                    else:
                        sample.light_ambient_color = light_ambient_color_
                    # light_diffuse_color
                    light_diffuse_color_ = dic[scenario_idx]["light"].get("diffuse_color")
                    if isinstance(light_diffuse_color_[0], list):
                        sample.light_diffuse_color = light_diffuse_color_[index]
                    else:
                        sample.light_diffuse_color = light_diffuse_color_
                    # light_specular_color
                    light_specular_color_ = dic[scenario_idx]["light"].get("specular_color")
                    if isinstance(light_specular_color_[0], list):
                        sample.light_specular_color = light_specular_color_[index]
                    else:
                        sample.light_specular_color = light_specular_color_
                    # light_location
                    light_location_ = dic[scenario_idx]["light"].get("location")
                    if isinstance(light_location_[0], list):
                        sample.light_location = light_location_[index]
                    else:
                        sample.light_location = light_location_
                    samples.append(sample)
        return samples, scenario_indexes

    @property
    def data_raw(self):
        return self._samples

    @property
    def train_data(self) -> List[Sample]:
        """Get train data."""
        if self._train_data is None or self._counter >= self._frequency:
            res = []
            data_by_location = copy.deepcopy(Sample.sort_by_scenario_location(self._samples))
            for _, sample_location_dict in data_by_location.items():
                for _, sample_list in sample_location_dict.items():
                    samples_ = []
                    # apply random rotation augmentation
                    if self._random_args["rotation"]["enable"]:
                        times = int(self._random_args["rotation"]["times"]) \
                            if self._random_args["rotation"]["times"] >= 1 else 1
                        angle_range = self._random_args["rotation"]["range"]
                        angle_list = [random.randint(0, (angle_range[1] - angle_range[0]) // times) +
                                      i * ((angle_range[1] - angle_range[0]) // times) +
                                      angle_range[0]
                                      for i in range(times)]
                        for idx in range(times):
                            sample = copy.deepcopy(sample_list[0])
                            sample.rotation[1] = angle_list[idx]
                            samples_.append(sample)
                    else:
                        samples_.extend(sample_list)
                    # apply random translate augmentation
                    if self._random_args["translate"]["enable"]:
                        lateral_range = self._random_args["translate"]["lateral"]
                        longitudinal_range = self._random_args["translate"]["longitudinal"]
                        for sample in samples_:
                            sample.location[0] += random.uniform(float(lateral_range[0]),
                                                                 float(lateral_range[1]))
                            sample.location[2] += random.uniform(float(longitudinal_range[0]),
                                                                 float(longitudinal_range[1]))
                    res.extend(samples_)
            self._counter = 0
            self._train_data = res
        self._counter += 1
        return copy.deepcopy(self._train_data)

    @property
    def eval_data(self) -> Dict[float, List[Sample]]:
        """Get eval data."""
        if self._eval_data is None:
            res = {}
            data_by_location = copy.deepcopy(Sample.sort_by_scenario_location(self._samples))
            for _, sample_location_dict in data_by_location.items():
                for _, sample_list in sample_location_dict.items():
                    if len(sample_list) > 1:
                        self._logger_console.warn("Multi rotation has defined in one location, something may wrong.")
                    angle_list = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
                    for angle in angle_list:
                        sample = copy.deepcopy(sample_list[0])
                        sample.rotation[1] = angle
                        if res.get(angle):
                            res.get(angle).append(sample)
                        else:
                            res[angle] = [sample]
            self._eval_data = res
        return copy.deepcopy(self._eval_data)

    @property
    def scenario_indexes(self):
        return self._scenario_indexes

