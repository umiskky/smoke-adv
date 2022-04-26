"""This script is used to eval."""
import argparse
import os.path as osp
import sys
import warnings

import numpy as np
import torch
from prettytable import PrettyTable
from tqdm import tqdm

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from dataset.vis_eval import vis2, vis
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

    # # Saving Metrics
    fd = osp.join(cfg.cfg_global["project_path"], "data/results/eval/random")

    # pipeline.stickers.adv_texture_hls = \
    #     "data/results/train/2022-04-21-10-34/visualization/adv_texture/00299_061.69341_adv_texture_hls.pth"
    pipeline.stickers.apply_gauss_perturb(-0.1, 0.1)

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
                box3d_branch_score_filtered = \
                    getattr(Loss, "_filter_with_threshold")(box3d_branch=box3d_branch_3d_radius_filtered,
                                                            threshold=0.25)
                box3d_branch_filter = box3d_branch_score_filtered

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
            # Vis
            pipeline.visualization.vis(scenario_index=sample.scenario_index,
                                       scenario=pipeline.scenario,
                                       renderer=pipeline.renderer,
                                       stickers=pipeline.stickers,
                                       smoke=pipeline.smoke,
                                       suffix="_angle-" + str(angle))
        t_scenario.close()
        t_angle.update()
    t_angle.close()
    delta_table = PrettyTable()
    delta_table.field_names = ["angle", "location_x", "location_y", "location_z",
                               "dimension_x", "dimension_y", "dimension_z", "iou_2d",
                               "car", "cyclist", "walker", "score",
                               "detection_rate", "attack_success_rate"]
    for angle, metrics in metrics_dict.items():
        # init
        location_list = []
        dimension_list = []
        iou_2d = []
        type_list = []
        score = []
        # process
        detection_rate = len(metrics) / len(dataset.get(angle))
        attack_success_rate = 1 - detection_rate
        if len(metrics) > 0:
            for metric in metrics:
                location_list.append(metric.location_delta)
                dimension_list.append(metric.dimension_delta)
                iou_2d.append(metric.iou_2d)
                type_list.append(metric.class_type)
                score.append(metric.score)
            np_location = np.array(location_list)
            np_location_mean = np.absolute(np_location).mean(axis=0)
            np_dimension = np.array(dimension_list)
            np_dimension_mean = np.absolute(np_dimension).mean(axis=0)
            np_iou_2d = np.array(iou_2d)
            np_type_list = np.array(type_list)
            np_score = np.array(score)
            # verbose
            delta_table.add_row([
                "%d" % angle,
                "%.3f" % np_location_mean[0], "%.3f" % np_location_mean[1], "%.3f" % np_location_mean[2],
                "%.3f" % np_dimension_mean[0], "%.3f" % np_dimension_mean[1], "%.3f" % np_dimension_mean[2],
                "%.3f" % np_iou_2d.mean(),
                "{:.2f}%".format((np_type_list == 0).sum().item() / len(metrics) * 100),
                "{:.2f}%".format((np_type_list == 1).sum().item() / len(metrics) * 100),
                "{:.2f}%".format((np_type_list == 2).sum().item() / len(metrics) * 100),
                "%.3f" % np_score.mean(),
                "{:.2f}%".format(detection_rate * 100),
                "{:.2f}%".format(attack_success_rate * 100)
            ])
            # plot
            path = osp.join(fd, "figure")
            makedirs(path)
            vis2(np_dimension, "Angle %d : Dimension" % angle, ['w', 'h', 'l'], "sample", "meters",
                 osp.join(path, "Angle_%d_Dimension.jpg" % angle))
            vis2(np_location, "Angle %d : Location" % angle, ['x', 'y', 'z'], "sample", "meters",
                 osp.join(path, "Angle_%d_Location.jpg" % angle))
            vis(np_score, "Angle %d : Score" % angle, "sample", "score",
                osp.join(path, "Angle_%d_Score.jpg" % angle))
            vis(np_iou_2d, "Angle %d : 2D IOU" % angle, "sample", "iou",
                osp.join(path, "Angle_%d_2D_IOU.jpg" % angle))
            vis(np_type_list, "Angle %d : Type" % angle, "sample", "type",
                osp.join(path, "Angle_%d_Type.jpg" % angle))
        else:
            # verbose
            delta_table.add_row([
                "%d" % angle,
                "", "", "",
                "", "", "",
                "", "", "",
                "", "",
                "{:.2f}%".format(detection_rate * 100),
                "{:.2f}%".format(attack_success_rate * 100)
            ])

    print(delta_table)


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


if __name__ == '__main__':
    args = parse_args()
    main_pipe(args)
