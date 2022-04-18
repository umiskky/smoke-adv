from typing import Union, Optional

import numpy as np


class Sample(object):
    def __init__(self, scenario_index: str) -> None:
        self._scenario_index = scenario_index
        self._K = None
        self._scale = 1.0
        self._location = [0.0, 0.0, 0.0]
        self._rotation = [0, 0, 0]
        self._light_ambient_color = [0.0, 0.0, 0.0]
        self._light_diffuse_color = [0.0, 0.0, 0.0]
        self._light_specular_color = [0.0, 0.0, 0.0]
        self._light_location = [0.0, 0.0, 0.0]

    # ===================== Getter =====================
    @property
    def scenario_index(self):
        return self._scenario_index

    @property
    def K(self):
        return self._K

    @property
    def K_inverse(self):
        return np.linalg.inv(self._K)

    @property
    def scale(self):
        return self._scale

    @property
    def location(self):
        return self._location

    @property
    def rotation(self):
        return self._rotation

    @property
    def light_ambient_color(self):
        return self._light_ambient_color

    @property
    def light_diffuse_color(self):
        return self._light_diffuse_color

    @property
    def light_specular_color(self):
        return self._light_specular_color

    @property
    def light_location(self):
        return self._light_location

    # ==================================================

    # ===================== Setter =====================
    @K.setter
    def K(self, K_):
        if isinstance(K_, np.ndarray) and K_.shape == (3, 3):
            self._K = K_
        else:
            raise ValueError("K must be ndarray type, and its shape is [3, 3].")

    @scale.setter
    def scale(self, scale_: Optional[Union[int, float]] = None):
        if scale_:
            self._scale = scale_

    @location.setter
    def location(self, location_: list):
        if isinstance(location_, list) and len(location_) == 3:
            self._location = [float(i) for i in location_]
        else:
            raise ValueError("The length of location list must be 3.")

    @rotation.setter
    def rotation(self, rotation_: Optional[list] = None):
        if rotation_:
            if isinstance(rotation_, list) and len(rotation_) == 3:
                self._rotation = [float(i) for i in rotation_]
            else:
                raise ValueError("The length of rotation list must be 3.")

    @light_ambient_color.setter
    def light_ambient_color(self, light_ambient_color_: Optional[list] = None):
        if light_ambient_color_:
            if isinstance(light_ambient_color_, list) and len(light_ambient_color_) == 3:
                self._light_ambient_color = [float(i) for i in self._color_formatter(light_ambient_color_)]
            else:
                raise ValueError("The length of light's ambient color list must be 3.")

    @light_diffuse_color.setter
    def light_diffuse_color(self, light_diffuse_color_: Optional[list] = None):
        if light_diffuse_color_:
            if isinstance(light_diffuse_color_, list) and len(light_diffuse_color_) == 3:
                self._light_diffuse_color = [float(i) for i in self._color_formatter(light_diffuse_color_)]
            else:
                raise ValueError("The length of light's diffuse color list must be 3.")

    @light_specular_color.setter
    def light_specular_color(self, light_specular_color_: Optional[list] = None):
        if light_specular_color_:
            if isinstance(light_specular_color_, list) and len(light_specular_color_) == 3:
                self._light_specular_color = [float(i) for i in self._color_formatter(light_specular_color_)]
            else:
                raise ValueError("The length of light's specular color list must be 3.")

    @light_location.setter
    def light_location(self, light_location_: Optional[list] = None):
        if light_location_:
            if isinstance(light_location_, list) and len(light_location_) == 3:
                self._light_location = [float(i) for i in light_location_]
            else:
                raise ValueError("The length of light's location list must be 3.")

    # ==================================================

    @classmethod
    def _color_formatter(cls, color: list):
        for idx in range(len(color)):
            if isinstance(color[idx], int):
                color[idx] /= 255.0
                color[idx] = cls._clip(color[idx])
        return color

    @staticmethod
    def _clip(num, min_=0.0, max_=1.0):
        return min(max(num, min_), max_)

    @staticmethod
    def sort_by_scenario(samples: list) -> dict:
        """
        Sort by scenario indexes. \n
        :param samples: list of Class Sample instances.
        :return: {scenario: [sample], ...}.
        """
        res = {}
        for sample in samples:
            if res.get(sample.scenario_index):
                res[sample.scenario_index].append(sample)
            else:
                res[sample.scenario_index] = [sample]
        return res

    @staticmethod
    def sort_by_scenario_location(samples: list) -> dict:
        """
        Sort by scenario locations. \n
        :param samples: list of Class Sample instances.
        :return: {scenario: {location: [sample]}, ...}.
        """
        res = {}
        samples_sort_by_scenario = Sample.sort_by_scenario(samples)
        for scenario_index, sample_list in samples_sort_by_scenario.items():
            for sample in sample_list:
                if res.get(scenario_index) and res.get(scenario_index).get(str(sample.location)):
                    res[scenario_index][str(sample.location)].append(sample)
                else:
                    res[scenario_index] = {str(sample.location): [sample]}
        return res
