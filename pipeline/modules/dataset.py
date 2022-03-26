import os
import os.path as osp

import codecs
import torch.utils.data as tud
import yaml

from smoke.waymo import getKMatrix


class Dataset(tud.Dataset):
    def __init__(self, args: dict) -> None:
        self._project_path = os.getenv("project_path")
        _, self._logger_console = args["logger"]
        if self._project_path is None:
            if self._logger_console is not None:
                self._logger_console.error("The Project Path in env is None.")
            raise ValueError('The Project Path in env is None.')
        self._meta_file = osp.join(self._project_path, args["meta"])
        self._data, self._scenario_indexes = self._load_meta_from_yaml(self._meta_file, osp.join(self._project_path, args["calib_dir"]))

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)

    @property
    def data(self):
        return self._data

    @property
    def scenario_indexes(self):
        return self._scenario_indexes

    @data.setter
    def data(self, data_):
        self._data = data_

    @staticmethod
    def _load_meta_from_yaml(meta_file: str, calib_path: str):
        # [scenario_idx, K, scale, rotation, translate(list), ambient_color, diffuse_color, specular_color, location]
        data = []
        indexes = []
        with codecs.open(meta_file, 'r', 'utf-8') as f:
            dic: dict = yaml.load(f, Loader=yaml.FullLoader)
        if len(dic) > 0:
            for scenario_idx in dic.keys():
                indexes.append(str(scenario_idx))
                K = getKMatrix(calib_idx=scenario_idx,
                               calib_path=calib_path)
                if dic[scenario_idx] is not None and dic[scenario_idx]["model_matrix"] is not None:
                    model_matrix_scale = dic[scenario_idx]["model_matrix"].get("scale")
                    model_matrix_rotation = dic[scenario_idx]["model_matrix"].get("rotation")
                    model_matrix_translate = dic[scenario_idx]["model_matrix"].get("translate")
                else:
                    model_matrix_scale = 0
                    model_matrix_rotation = [0, 0, 0]
                    model_matrix_translate = [[0, 0, 0]]
                if dic[scenario_idx] is not None and dic[scenario_idx]["light"] is not None:
                    light_ambient_color = dic[scenario_idx]["light"].get("ambient_color")
                    light_diffuse_color = dic[scenario_idx]["light"].get("diffuse_color")
                    light_specular_color = dic[scenario_idx]["light"].get("specular_color")
                    light_location = dic[scenario_idx]["light"].get("location")
                else:
                    light_ambient_color = light_diffuse_color = light_specular_color = light_location = [0, 0, 0]

                item_num = len(model_matrix_translate)
                # ============================ check valid or not ============================
                assert isinstance(model_matrix_translate, list) \
                       and item_num > 0 \
                       and isinstance(model_matrix_translate[0], list)

                def check(value, length) -> bool:
                    assert isinstance(value, list) and len(value) > 0
                    if isinstance(value[0], list):
                        return len(value) == length
                    return True

                assert check(light_ambient_color, item_num) \
                       and check(light_diffuse_color, item_num) \
                       and check(light_specular_color, item_num) \
                       and check(light_location, item_num)

                if isinstance(model_matrix_scale, list):
                    assert check(model_matrix_scale, item_num)

                if model_matrix_rotation is not None:
                    assert check(model_matrix_rotation, item_num)
                # ============================================================================

                for idx in range(item_num):
                    item = [str(scenario_idx), K]
                    if isinstance(model_matrix_scale, list):
                        item.append(model_matrix_scale[idx])
                    else:
                        item.append(model_matrix_scale)

                    if model_matrix_rotation is not None and isinstance(model_matrix_rotation[0], list):
                        item.append(model_matrix_rotation[idx])
                    else:
                        item.append(model_matrix_rotation)

                    item.append(model_matrix_translate[idx])

                    if isinstance(light_ambient_color[0], list):
                        item.append(light_ambient_color[idx])
                    else:
                        item.append(light_ambient_color)

                    if isinstance(light_diffuse_color[0], list):
                        item.append(light_diffuse_color[idx])
                    else:
                        item.append(light_diffuse_color)

                    if isinstance(light_specular_color[0], list):
                        item.append(light_specular_color[idx])
                    else:
                        item.append(light_specular_color)

                    if isinstance(light_location[0], list):
                        item.append(light_location[idx])
                    else:
                        item.append(light_location)

                    data.append(item)
        return data, indexes


if __name__ == '__main__':
    from tools.config import Config
    from tools.logger import Logger

    cfg = Config("../../data/config/attack.yaml")
    logger = Logger(cfg.cfg_logger)
    cfg.cfg_all = logger.broadcast_logger(cfg.cfg_all)
    dataset = Dataset(cfg.cfg_dataset)
    pass
