import codecs
import os

import yaml


class Config:
    def __init__(self, path: str) -> None:
        if not path:
            raise ValueError('Please specify the configuration file path.')

        if not os.path.exists(path):
            raise FileNotFoundError('File {} does not exist'.format(path))

        if path.endswith('yml') or path.endswith('yaml'):
            self.dic = self._parse_from_yaml(path)
        else:
            raise RuntimeError('Config file should in yaml format!')

        self._setting_device()

    def _parse_from_yaml(self, path: str):
        """Parse a yaml file and build config"""
        with codecs.open(path, 'r', 'utf-8') as file:
            dic = yaml.load(file, Loader=yaml.FullLoader)
        return dic

    def _setting_device(self):
        """Setting all config device"""
        exclude = ["global", "visualization"]
        for key in self.dic.keys():
            if key not in exclude and "device" not in self.dic[key].keys():
                self.dic[key]["device"] = self.dic["global"]["device"]

    @property
    def cfg_global(self):
        return self.dic["global"]

    @property
    def cfg_object(self):
        return self.dic["object"]

    @property
    def cfg_stickers(self):
        return self.dic["stickers"]

    @property
    def cfg_scenario(self):
        return self.dic["scenario"]

    @property
    def cfg_renderer(self):
        return self.dic["renderer"]

    @property
    def cfg_defense(self):
        return self.dic["defense"]

    @property
    def cfg_smoke(self):
        return self.dic["smoke"]

    @property
    def cfg_attack(self):
        return self.dic["attack"]

    @property
    def cfg_visualization(self):
        return self.dic["visualization"]
