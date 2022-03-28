import argparse
import os
import os.path as osp

import torch

from pipeline.modules.early_stop import EarlyStop
from pipeline.modules.pgd_optimizer import PGDOptimizer
from pipeline.pipeline import Pipeline
from tools.config import Config
from tools.debug_utils import time_block
from tools.logger import Logger


def parse_args():
    parser = argparse.ArgumentParser(description='Attack Pipeline')

    # params of evaluate
    parser.add_argument(
        "--config",
        "-f",
        dest="cfg",
        default="./data/config/attack.yaml",
        help="The config file path.",
        required=False,
        type=str)
    return parser.parse_args()


def main(args):
    cfg = Config(args.cfg)
    logger = Logger(cfg.cfg_logger)
    logger.broadcast_logger(cfg.cfg_all, exclude=[])
    logger.logger_comet.log_parameters(logger.encode_config_for_log(cfg.cfg_for_log))
    workdir = os.getenv("project_path")
    # log meta.yaml & attack.yaml
    logger.logger_comet.log_asset(osp.join(workdir, cfg.cfg_dataset["meta"]))
    logger.logger_comet.log_asset(osp.join(workdir, "data/config/attack.yaml"))

    pipeline = Pipeline(cfg)

    es = EarlyStop(max_step=40)
    train_flag = True
    epoch = 0
    step = 0
    step_loss_list = []

    pgd = PGDOptimizer(params=[pipeline.stickers.patch],
                       alpha=cfg.cfg_attack["optimizer"]["alpha"],
                       clip_min=cfg.cfg_attack["optimizer"]["clip_min"],
                       clip_max=cfg.cfg_attack["optimizer"]["clip_max"],
                       position=cfg.cfg_stickers["position"],
                       size=cfg.cfg_stickers["size"],
                       device=cfg.cfg_global["device"])

    while train_flag:
        _epoch_loss = 0
        for _, data in enumerate(pipeline.dataset.data):
            try:
                with time_block("Forward & Backward & Step"):
                    loss = pipeline.forward(data)
                    if loss is not None:
                        loss.backward()
                    pgd.record()
            except KeyboardInterrupt:
                print("Stop Attack Manually!")
                logger.close_logger()

            with torch.no_grad():
                if loss is None:
                    loss = torch.tensor(0.0)
                _step_loss = loss.clone().cpu().item() * -1
                _epoch_loss += _step_loss
                step_loss_list.append(_step_loss)
                # # Visualization Pipeline
                with time_block("Vis"):
                    pipeline.visualization.vis(scenario_index=data[0],
                                               epoch=epoch,
                                               step=step,
                                               scenario=pipeline.scenario,
                                               renderer=pipeline.renderer,
                                               stickers=pipeline.stickers,
                                               smoke=pipeline.smoke)
                # print to terminal
                print("epoch: %04d" % epoch + "   " + "step: %04d" % step + "   " + "step_loss: %.10f" % _step_loss)
                # log
                logger.logger_comet.log_metric("loss_step", _step_loss, step=step)
                # clear and prepare for the next step
                step += 1
        # update patch
        pgd.step(step_type="softmax", step_loss_list=step_loss_list)
        # print to terminal
        print("==============================================================")
        print("epoch: %04d" % epoch +
              "   epoch_loss: %.10f" % _epoch_loss +
              "   mean_loss: %.10f" % (_epoch_loss/len(pipeline.dataset.data)))
        print("==============================================================\n")

        # log
        logger.logger_comet.log_metric("loss_epoch", _epoch_loss, epoch=epoch)
        logger.logger_comet.log_metric("loss_epoch_mean", _epoch_loss/len(pipeline.dataset.data), epoch=epoch)
        # save patch
        pipeline.visualization.save_patch(epoch, _epoch_loss, pipeline.stickers)
        # eval perturbation norm
        pipeline.visualization.eval_norm(epoch, pipeline.object_loader, pipeline.stickers)
        # check if you can stop training
        if _epoch_loss <= cfg.cfg_attack["target_score"]:
            train_flag = False
        else:
            train_flag = es.step(_epoch_loss)

        # save patch before exit
        if not train_flag:
            # save patch
            pipeline.visualization.save_patch(epoch, _epoch_loss, pipeline.stickers)

        # clear and prepare for the next epoch
        _epoch_loss = 0
        step_loss_list = []
        epoch += 1

    # ensure all metrics and code are logged before exiting
    logger.close_logger()


if __name__ == '__main__':
    args = parse_args()
    # with torch.autograd.set_detect_anomaly(True):
    main(args)
