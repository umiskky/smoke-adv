from pipeline.pipeline import Pipeline
from tools.config import Config
from tools.logger import Logger


def main_pipe(args):
    cfg = Config(args.cfg)
    logger = Logger(cfg.cfg_logger)
    logger.broadcast_logger(cfg.cfg_all, exclude=[])
    pipeline = Pipeline(cfg)

    for _, data in enumerate(pipeline.dataset.data):
        try:
            pipeline.forward(data)
            # Visualization Pipeline
            pipeline.visualization.vis(scenario_index=data[0],
                                       epoch=0,
                                       step=0,
                                       scenario=pipeline.scenario,
                                       renderer=pipeline.renderer,
                                       stickers=pipeline.stickers,
                                       smoke=pipeline.smoke)
        except KeyboardInterrupt:
            print("Stop Attack Manually!")
            logger.close_logger()
        logger.close_logger()
