import argparse

import torch

from attack.attack import Attack
from attack.pgd_optimizer import PGDOptimizer
from tools.config import Config
from tools.early_stop import EarlyStop
from tools.utils import state_saving


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
    attack = Attack(cfg)

    if attack.loss is not None:
        pgd = PGDOptimizer(params=[attack.stickers.patch],
                           alpha=cfg.cfg_attack["optimizer"]["alpha"],
                           clip_min=cfg.cfg_attack["optimizer"]["clip_min"],
                           clip_max=cfg.cfg_attack["optimizer"]["clip_max"],
                           position=cfg.cfg_stickers["position"],
                           size=cfg.cfg_stickers["size"])
        es = EarlyStop()
        flag = True
        epoch = 0
        while flag:
            try:
                pgd.zero_grad()
                loss = attack.forward()
                loss.backward()
                pgd.step()
            except KeyboardInterrupt:
                print("Stop Attack Manually!")
                break

            with torch.no_grad():
                _loss = loss.clone().cpu().item() * -1
                _patch = attack.stickers.patch.detach().clone().cpu()
                flag = es.step(_loss)
                # save texture patch
                state_dict = {"patch": _patch, "epoch": epoch, "loss": _loss, "cfg": cfg}
                state_saving(state=state_dict, epoch=epoch, loss=loss, path=attack.visualization.experiment_path)
                # print to terminal
                print("epoch: %05d" % epoch + "   " + "loss: %.5f" % _loss)
                epoch += 1
            try:
                pass
            except KeyboardInterrupt:
                print("Stop Attack Manually!")
                flag = False

    pass


if __name__ == '__main__':
    args = parse_args()
    with torch.autograd.set_detect_anomaly(True):
        main(args)
