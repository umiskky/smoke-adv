import argparse

import torch

from attack.attack import Attack
from attack.pgd_optimizer import PGDOptimizer
from tools.config import Config
from tools.early_stop import EarlyStop
from tools.timer import time_block
from tools.utils import state_saving


def parse_args():
    parser = argparse.ArgumentParser(description='Attack Pipeline')

    # params of evaluate
    parser.add_argument(
        "--config",
        "-f",
        dest="cfg",
        default="./data/config/attack_patch.yaml",
        # default="./data/config/render.yaml",
        # default="./data/config/attack.yaml",
        help="The config file path.",
        required=False,
        type=str)
    return parser.parse_args()


def main(args):
    cfg = Config(args.cfg)
    logger = cfg.logger
    attack = Attack(cfg)

    if attack.loss is not None and cfg.cfg_attack["enable"]:
        pgd = PGDOptimizer(params=[attack.stickers.patch],
                           alpha=cfg.cfg_attack["optimizer"]["alpha"],
                           clip_min=cfg.cfg_attack["optimizer"]["clip_min"],
                           clip_max=cfg.cfg_attack["optimizer"]["clip_max"],
                           position=cfg.cfg_stickers["position"],
                           size=cfg.cfg_stickers["size"])
        es = EarlyStop(max_step=60)
        flag = True
        epoch = 0
        while flag:
            try:
                with time_block("Forward"):
                    pgd.zero_grad()
                    loss = attack.forward()
                with time_block("Backward & Step"):
                    loss.backward()
                    pgd.step()
            except KeyboardInterrupt:
                print("Stop Attack Manually!")
                cfg.close_logger()
                break

            with torch.no_grad():
                _loss = loss.clone().cpu().item() * -1
                _patch = attack.stickers.patch.detach().clone().cpu().numpy()
                # Visualization Pipeline
                attack.visualization.counter = epoch
                with time_block("Vis"):
                    attack.visualization.vis(scenario=attack.scenario,
                                             renderer=attack.renderer,
                                             stickers=attack.stickers,
                                             smoke=attack.smoke)
                # TODO
                # save texture patch
                if cfg.cfg_attack["enable"] and cfg.cfg_attack["save"]:
                    state_dict = {"patch": _patch, "epoch": epoch, "loss": _loss}
                    state_saving(state=state_dict, epoch=epoch, loss=_loss, path=attack.visualization.experiment_path)
                # print to terminal
                print("epoch: %05d" % epoch + "   " + "loss: %.5f" % _loss)
                logger.log_metric("loss", _loss, step=epoch)
                # Stop Train
                if _loss <= 0:
                    flag = False
                else:
                    flag = es.step(_loss)
                epoch += 1
            try:
                pass
            except KeyboardInterrupt:
                print("Stop Attack Manually!")
                cfg.close_logger()
                flag = False
    else:
        attack.forward()
        attack.visualization.vis(scenario=attack.scenario,
                                 renderer=attack.renderer,
                                 stickers=attack.stickers,
                                 smoke=attack.smoke)

    # ensure all metrics and code are logged before exiting
    cfg.close_logger()
    pass


if __name__ == '__main__':
    args = parse_args()
    # with torch.autograd.set_detect_anomaly(True):
    main(args)
