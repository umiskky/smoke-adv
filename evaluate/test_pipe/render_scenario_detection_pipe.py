import argparse

import evaluate.test_pipe.base_pipe as tpp


def parse_args():
    parser = argparse.ArgumentParser(description='Attack Pipeline')

    # params of evaluate
    parser.add_argument(
        "--config",
        "-f",
        dest="cfg",
        default="./render_scenario_detection_pipe.yaml",
        help="The config file path.",
        required=False,
        type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    tpp.main_pipe(args)
