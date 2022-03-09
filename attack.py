import argparse

from attack.attack import Attack
from tools.config import Config


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
    attack.forward()
    pass


if __name__ == '__main__':
    args = parse_args()
    main(args)
