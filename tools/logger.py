import os
import os.path as osp
import logging
from comet_ml import Experiment, OfflineExperiment


class Logger:
    _level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self, args: dict) -> None:
        self._timestamp = os.getenv("timestamp")
        self._project_path = os.getenv("project_path")
        self._logger_comet = None
        self._logger_console = None
        if args["comet"]["enable"]:
            if "comet_online" == args["comet"]["type"]:
                self._init_logger_online(args)
            elif "comet_offline" == args["comet"]["type"]:
                self._init_logger_offline(args, self._project_path)
        if args["logging"]["enable"]:
            self._init_logger_console(args)

        if self._logger_comet is not None:
            self._logger_comet.set_name(self._timestamp)

    def _init_logger_online(self, args: dict):
        args = args["comet"]
        self._logger_comet = Experiment(
            api_key=args["api_key"],
            project_name=args["project_name"],
            workspace=args["workspace"],
            disabled=args["test"],
            log_env_details=True,
            log_env_gpu=True,
            log_env_cpu=True,
            log_env_host=True,
            log_git_metadata=False,
            log_git_patch=False,
            auto_param_logging=False,
        )

    def _init_logger_offline(self, args: dict, project_path: str):
        args = args["comet"]
        self._logger_comet = OfflineExperiment(
            project_name=args["project_name"],
            workspace=args["workspace"],
            offline_directory=osp.join(project_path,
                                       args["offline_dir"],
                                       self._timestamp),
            disabled=args["test"],
            log_env_details=True,
            log_env_gpu=True,
            log_env_cpu=True,
            log_env_host=True,
            log_git_metadata=False,
            log_git_patch=False,
            auto_param_logging=False,
        )

    def _init_logger_console(self, args: dict):
        args = args["logging"]
        logger = logging.getLogger(self._timestamp)
        logger.setLevel(self._level_relations[args["level"]])
        # create console handler
        consoleHandler = logging.StreamHandler()
        consoleHandler.setLevel(self._level_relations[args["level"]])
        # set formatter
        formatter = logging.Formatter(args["format"],
                                      datefmt='%Y-%m-%d %H:%M:%S')
        consoleHandler.setFormatter(formatter)
        # add handler
        logger.addHandler(consoleHandler)
        self._logger_console = logger

    def broadcast_logger(self, config: dict, exclude=["logger"]) -> dict:
        config = config.copy()
        for key in config.keys():
            if key not in exclude and "logger" not in config[key].keys():
                config[key]["logger"] = (self._logger_comet, self._logger_console)
        return config

    def close_logger(self):
        if self._logger_comet is not None:
            self._logger_comet.end()
        if self._logger_console is not None:
            logging.shutdown()

    @property
    def logger_comet(self):
        return self._logger_comet

    @property
    def logger_console(self):
        return self._logger_console

    @staticmethod
    def encode_config_for_log(config: dict, character=".") -> dict:
        merge_dict = {}
        for key, value in config.items():
            if isinstance(value, dict):
                tmp_dict = Logger.encode_config_for_log(value)
                for key_, value_ in tmp_dict.items():
                    merge_dict[str(key) + character + str(key_)] = value_
            else:
                merge_dict[key] = value
        return merge_dict
