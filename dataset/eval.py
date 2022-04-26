"""This script is used to eval dataset."""
import argparse
import os.path as osp
import pickle
import sys
import warnings

import torch
from tqdm import tqdm

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from dataset.metric import Metric
from pipeline.modules.loss import Loss
from pipeline.pipeline import Pipeline
from tools.config import Config
from tools.file_utils import makedirs
from tools.logger import Logger

warnings.filterwarnings("ignore")


def main_pipe(args):
    cfg = Config(args.cfg)
    logger = Logger(cfg.cfg_logger)
    logger.broadcast_logger(cfg.cfg_all, exclude=[])
    pipeline = Pipeline(cfg)

    dataset = pipeline.dataset.eval_data
    step = 0
    metrics_dict = {}
    t_angle = tqdm(total=len(dataset.keys()))
    for angle, samples in dataset.items():
        t_scenario = tqdm(total=len(samples))
        # init for metrics dict
        if metrics_dict.get(angle) is None:
            metrics_dict[angle] = []
        for sample in samples:
            metric = Metric()
            metric.sample = sample
            try:
                box3d_branch, box_pseudo_gt = pipeline.forward(sample)
                box3d_branch_target_filtered = getattr(Loss, "_filter_with_target")(box3d_branch=box3d_branch,
                                                                                    targets=[1, 2])
                box3d_branch_3d_radius_filtered = \
                    getattr(Loss, "_filter_with_3d_radius")(box3d_branch=box3d_branch_target_filtered,
                                                            box_3d_gt=box_pseudo_gt.get("3d"),
                                                            radius=2)
                box3d_branch_filter = box3d_branch_3d_radius_filtered

                # Record GT
                metric.dimension_gt = cfg.cfg_object.get("size")
                metric.box_2d_gt = box_pseudo_gt.get("2d")

                if box3d_branch_filter is not None and box3d_branch_filter.shape[0] > 0:
                    # sort detection results by score
                    _, indices = torch.sort(box3d_branch_filter[:, -1], dim=0, descending=True, stable=True)
                    indices = torch.flatten(indices)
                    # Get max score item
                    prediction = box3d_branch_filter[indices[0], :]
                    # Record Prediction
                    metric.class_type = prediction[0].int().item()
                    metric.score = prediction[-1].item()
                    metric.dimension_pred = prediction[6:9].roll(shifts=1, dims=0).cpu().numpy().tolist()
                    location = prediction[9:12]
                    location[1] -= float(cfg.cfg_renderer["camera"]["height"])
                    metric.location_pred = location.cpu().numpy().tolist()
                    metric.box_2d_pred = prediction[2:6].cpu().numpy().tolist()
                else:
                    print("Scenario " + str(sample.scenario_index) + " failed to detection in angle " + str(angle))
                metrics_dict[angle].append(metric)
                t_scenario.update()
            except KeyboardInterrupt:
                print("Stop Attack Manually!")
                logger.close_logger()
                t_scenario.close()
                t_angle.close()
                sys.exit(0)
            logger.close_logger()
            step += 1
        t_scenario.close()
        t_angle.update()
    t_angle.close()

    # Saving Metrics
    fd = osp.join(cfg.cfg_global["project_path"], "data/results/eval/dataset_train")
    makedirs(fd)
    fp = osp.join(fd, "metrics_eval_dataset.pickle")
    with open(fp, 'wb') as fpb:
        pickle.dump(metrics_dict, fpb)


def parse_args():
    parser = argparse.ArgumentParser(description='Attack Pipeline')

    # params of evaluate
    parser.add_argument(
        "--config",
        "-f",
        dest="cfg",
        default="./render_config.yaml",
        help="The config file path.",
        required=False,
        type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main_pipe(args)
