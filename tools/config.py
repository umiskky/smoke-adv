import codecs
import os
import random

import numpy as np
import torch
import yaml

from tools.file_utils import get_utc8_time


class Config:
    def __init__(self, path: str) -> None:
        if not path:
            raise ValueError('Please specify the configuration file path.')

        if not os.path.exists(path):
            raise FileNotFoundError('File {} does not exist'.format(path))

        if path.endswith('yml') or path.endswith('yaml'):
            self._dic = self._parse_from_yaml(path)
            self._dict_raw = self._parse_from_yaml(path, False)
        else:
            raise RuntimeError('Config file should in yaml format!')

        self._timestamp = get_utc8_time()
        self._broadcast_cfg()
        self._setting_seed()

    def _setting_env(self):
        os.environ["timestamp"] = self._timestamp
        os.environ["project_path"] = self._dic["global"]["project_path"]
        os.environ["debug"] = "True" if self._dic["global"]["debug"] else ""

    def _broadcast_cfg(self):
        """Broadcast Config File Settings & Setting Environment Variables"""
        exclude = ["global", "logger"]
        for key in self._dic.keys():
            if key not in exclude and "device" not in self._dic[key].keys():
                self._dic[key]["device"] = self._dic["global"]["device"]
        self._dic["stickers"]["clip_min"] = self._dic["attack"]["optimizer"]["clip_min"]
        self._dic["stickers"]["clip_max"] = self._dic["attack"]["optimizer"]["clip_max"]
        self._setting_env()

    def _setting_seed(self):
        seed = self._dic["global"]["seed"]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        # torch.use_deterministic_algorithms(True)

    @staticmethod
    def _parse_from_yaml(path: str, full=True):
        """Parse a yaml file and build config"""

        with codecs.open(path, 'r', 'utf-8') as file:
            dic = yaml.load(file, Loader=yaml.FullLoader)
        if full and 'base' in dic:
            project_dir = dic["global"]["project_path"]
            base_path = dic.pop('base')
            base_path = os.path.join(project_dir, base_path)
            base_dic = Config._parse_from_yaml(base_path)
            dic = Config._update_dic(dic, base_dic)
        return dic

    @staticmethod
    def _update_dic(dic, base_dic):
        """Update config from dic based base_dic"""
        base_dic = base_dic.copy()
        for key, val in dic.items():
            if isinstance(val, dict) and key in base_dic:
                base_dic[key] = Config._update_dic(val, base_dic[key])
            else:
                base_dic[key] = val
        dic = base_dic
        return dic

    @property
    def cfg_for_log(self):
        return self._dict_raw

    @property
    def cfg_all(self):
        return self._dic

    @cfg_all.setter
    def cfg_all(self, new_dict: dict):
        self._dic = new_dict

    @property
    def cfg_global(self):
        return self._dic["global"]

    @property
    def cfg_enable(self):
        return self._dic["enable"]

    @property
    def cfg_dataset(self):
        return self._dic["dataset"]

    @property
    def cfg_object(self):
        return self._dic["object"]

    @property
    def cfg_stickers(self):
        return self._dic["stickers"]

    @property
    def cfg_scenario(self):
        return self._dic["scenario"]

    @property
    def cfg_renderer(self):
        return self._dic["renderer"]

    @property
    def cfg_defense(self):
        return self._dic["defense"]

    @property
    def cfg_smoke(self):
        return self._dic["smoke"]

    @property
    def cfg_attack(self):
        return self._dic["attack"]

    @property
    def cfg_logger(self):
        return self._dic["logger"]

    @property
    def cfg_eval(self):
        return self._dic["eval"]

    def __str__(self):
        return yaml.dump(self._dic)
