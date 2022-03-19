import codecs
import os
# import comet_ml.scripts.comet_upload
import yaml
import os.path as osp
from comet_ml import Experiment, OfflineExperiment

from tools.utils import get_utc8_time


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

        self.timestamp = get_utc8_time()

        self._logger = None
        if self.dic["logger"]["enable"]:
            self._logger = self.init_logger(self.dic["logger"]["test_pipeline"])
            self._logger.log_parameters(self.encode_config_for_log(self.dic))
        self._setting_device()
        self._setting_logger()
        self._setting_timestamp()

    def init_logger(self, debug=False) -> Experiment:
        logger_cfg = self.dic["logger"]
        if "comet_online" == logger_cfg["type"]:
            self._logger = Experiment(
                api_key=logger_cfg["api_key"],
                project_name=logger_cfg["project_name"],
                workspace=logger_cfg["workspace"],
                disabled=debug,
                log_env_details=True,
                log_env_gpu=True,
                log_env_cpu=True,
                log_env_host=True,
                log_git_metadata=False,
                log_git_patch=False,
                auto_param_logging=False,
                # experiment_key=self.timestamp + "_" + self.timestamp,
            )
        elif "comet_offline" == logger_cfg["type"]:
            self._logger = OfflineExperiment(
                # api_key=logger_cfg["api_key"],
                project_name=logger_cfg["project_name"],
                workspace=logger_cfg["workspace"],
                offline_directory=osp.join(self.dic["global"]["project_path"],
                                           logger_cfg["offline_dir"],
                                           self.timestamp),
                disabled=debug,
                log_env_details=True,
                log_env_gpu=True,
                log_env_cpu=True,
                log_env_host=True,
                log_git_metadata=False,
                log_git_patch=False,
                auto_param_logging=False,
                # experiment_key=self.timestamp + "_" + self.timestamp,
            )
        if self._logger is not None:
            self._logger.set_name(self.timestamp)
        return self._logger

    def close_logger(self):
        if self._logger is not None:
            self._logger.end()

    def _setting_device(self):
        """Setting all config device"""
        exclude = ["global", "visualization"]
        for key in self.dic.keys():
            if key not in exclude and "device" not in self.dic[key].keys():
                self.dic[key]["device"] = self.dic["global"]["device"]

    def _setting_logger(self):
        """Setting all config logger"""
        exclude = ["global"]
        for key in self.dic.keys():
            if key not in exclude and "logger" not in self.dic[key].keys():
                self.dic[key]["logger"] = self._logger

    def _setting_timestamp(self):
        """Setting all timestamp"""
        exclude = ["object", "stickers", "scenario", "renderer", "defense", "smoke", "attack", "logger"]
        for key in self.dic.keys():
            if key not in exclude and "timestamp" not in self.dic[key].keys():
                self.dic[key]["timestamp"] = self.timestamp

    @staticmethod
    def encode_config_for_log(config: dict, character=".") -> dict:
        merge_dict = {}
        for key, value in config.items():
            if isinstance(value, dict):
                tmp_dict = Config.encode_config_for_log(value)
                for key_, value_ in tmp_dict.items():
                    merge_dict[str(key) + character + str(key_)] = value_
            else:
                merge_dict[key] = value
        return merge_dict

    @staticmethod
    def _parse_from_yaml(path: str):
        """Parse a yaml file and build config"""
        with codecs.open(path, 'r', 'utf-8') as file:
            dic = yaml.load(file, Loader=yaml.FullLoader)
        return dic

    @property
    def logger(self) -> Experiment:
        return self._logger

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

    @property
    def cfg_logger(self):
        return self.dic["logger"]
