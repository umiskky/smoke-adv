import argparse

import torch

from pipeline.pipeline import Pipeline
from tools.config import Config
from tools.logger import Logger


def parse_args():
    parser = argparse.ArgumentParser(description='Attack Pipeline')

    # params of evaluate
    parser.add_argument(
        "--config",
        "-f",
        dest="cfg",
        default="../data/config/eval.yaml",
        help="The config file path.",
        required=False,
        type=str)
    return parser.parse_args()


def main(args):
    cfg = Config(args.cfg)

    logger = Logger(cfg.cfg_logger)
    logger.broadcast_logger(cfg.cfg_all, exclude=[])

    pipeline = Pipeline(cfg)
    pipeline.stickers.patch = "data/results/2022-03-28-17-07/visualization/patch/00610_000.18310_patch.pth"

    epoch = 0
    step = 0
    _epoch_loss = 0
    loss = None

    # Sync visualization epoch
    pipeline.visualization.epoch = epoch
    for _, data in enumerate(pipeline.dataset.data):
        # Sync visualization step
        pipeline.visualization.step = step
        try:
            loss = pipeline.forward(data)
        except KeyboardInterrupt:
            print("Stop Attack Manually!")
            logger.close_logger()

        if loss is None:
            loss = torch.tensor(0)
        _step_loss = loss.clone().cpu().item() * -1
        _epoch_loss += _step_loss
        # # Visualization Pipeline
        pipeline.visualization.vis(scenario_index=data[0],
                                   scenario=pipeline.scenario,
                                   renderer=pipeline.renderer,
                                   stickers=pipeline.stickers,
                                   smoke=pipeline.smoke)
        # print to terminal
        print("step: %04d" % step + "   " + "step_loss: %.10f" % _step_loss)
        # log
        logger.logger_comet.log_metric("loss_step", _step_loss, step=step)
        # eval perturbation norm
        pipeline.visualization.eval_norm(step, pipeline.object_loader, pipeline.stickers)
        # clear and prepare for the next step
        step += 1
    # print to terminal
    print("==============================================================")
    print("epoch: %04d" % epoch +
          "   epoch_loss: %.10f" % _epoch_loss +
          "   mean_loss: %.10f" % (_epoch_loss / len(pipeline.dataset.data)))
    print("==============================================================\n")

    # log
    logger.logger_comet.log_metric("loss_epoch", _epoch_loss, epoch=epoch)
    logger.logger_comet.log_metric("loss_epoch_mean", _epoch_loss / len(pipeline.dataset.data), epoch=epoch)

    # ensure all metrics and code are logged before exiting
    logger.close_logger()


if __name__ == '__main__':
    args = parse_args()
    # with torch.autograd.set_detect_anomaly(True):
    main(args)
