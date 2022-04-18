"""This script is used to make dataset."""
import argparse

import torch
from prettytable import PrettyTable

from dataset.color_temperature import ColorTemperature
from pipeline.modules.loss import Loss
from pipeline.pipeline import Pipeline
from smoke.obstacle import Obstacle
from tools.config import Config
from tools.logger import Logger


def main_pipe(args):
    cfg = Config(args.cfg)
    logger = Logger(cfg.cfg_logger)
    logger.broadcast_logger(cfg.cfg_all, exclude=[])
    pipeline = Pipeline(cfg)

    # dataset = pipeline.dataset.generate()
    dataset = pipeline.dataset.data_raw
    step = 0
    pipeline.visualization.epoch = 0
    for _, sample in enumerate(dataset):

        # ========================== Verbose 1 ==========================
        # Color Temperature Table
        color_table1 = PrettyTable()
        color_table2 = PrettyTable()
        color_table1.field_names = ["~10%", "~20%", "~30%", "~40%", "~50%"]
        color_table2.field_names = ["~60%", "~70%", "~80%", "~90%"]
        phi_list = [0.22, 0.04, -0.14, -0.32, -0.5, -0.68, -0.86, -1.04, -1.22]
        temperature_list = []
        scenario, _ = pipeline.scenario.forward(sample.scenario_index)
        for phi in phi_list:
            color_t = ColorTemperature(phi=phi)
            temperature = color_t.calculate_temperature(ColorTemperature.transform_image(scenario))
            temperature_list.append(temperature)
        color_table1.add_row(temperature_list[:5])
        color_table2.add_row(temperature_list[5:])
        print(color_table1)
        print(color_table2)
        # ===============================================================

        pipeline.visualization.step = step
        try:
            box3d_branch, box_pseudo_gt = pipeline.forward(sample)
            box3d_branch_target_filtered = getattr(Loss, "_filter_with_target")(box3d_branch=box3d_branch,
                                                                                targets=[1, 2])
            box3d_branch_3d_radius_filtered = getattr(Loss, "_filter_with_3d_radius")(box3d_branch=box3d_branch_target_filtered,
                                                                                      box_3d_gt=box_pseudo_gt.get("3d"),
                                                                                      radius=2)
            box3d_branch_filter = box3d_branch_3d_radius_filtered

            # ========================== Verbose 2 ==========================
            if box3d_branch_filter is None:
                print("{0} scenario false to detection!".format(sample.scenario_index))
                # Visualization Pipeline
                pipeline.visualization.vis(scenario_index=sample.scenario_index,
                                           scenario=pipeline.scenario,
                                           renderer=pipeline.renderer,
                                           stickers=pipeline.stickers,
                                           smoke=pipeline.smoke)
                continue
            box3d_branch_filter = box3d_branch_filter.clone()
            # sort detection results by score
            _, indices = torch.sort(box3d_branch_filter[:, -1], dim=0, descending=True, stable=True)
            indices = torch.flatten(indices)
            # print title
            print("%s scenario:" % sample.scenario_index)
            table = PrettyTable()
            table.field_names = ["Index", "Type", "Width", "Height", "Length",
                                 "Location_x", "Location_y", "Location_z", "Score"]
            # print gt
            dimension_gt = box_pseudo_gt.get("3d").get("dimensions")
            location_gt = box_pseudo_gt.get("3d").get("location")
            table.add_row(['', 'GT', '%.3f' % dimension_gt[0], '%.3f' % dimension_gt[1], '%.3f' % dimension_gt[2],
                           '%.3f' % location_gt[0], '%.3f' % location_gt[1], '%.3f' % location_gt[2], ''])
            for index in range(indices.shape[0]):
                prediction = box3d_branch_filter[indices[index], :]
                type_ = Obstacle.type_map.get(prediction[0].int().item())
                dimensions = prediction[6:9].roll(shifts=1, dims=0)
                location = prediction[9:12]
                score = prediction[-1]
                # print prediction
                table.add_row([index, type_, '%.3f' % dimensions[0].item(), '%.3f' % dimensions[1].item(),
                               '%.3f' % dimensions[2].item(), '%.3f' % location[0].item(), '%.3f' % location[1].item(),
                               '%.3f' % location[2].item(), '%.3f' % score])
                # print delta between prediction and gt
                table.add_row(['', '',
                               '%.3f' % (dimensions[0].item() - dimension_gt[0]),
                               '%.3f' % (dimensions[1].item() - dimension_gt[1]),
                               '%.3f' % (dimensions[2].item() - dimension_gt[2]),
                               '%.3f' % (location[0].item() - location_gt[0]),
                               '%.3f' % (location[1].item() - location_gt[1]),
                               '%.3f' % (location[2].item() - location_gt[2]),
                               ''])
            print(table)
            # Change Vis content
            pipeline.smoke.visualization["detection"] = box3d_branch_filter.cpu().clone()
            # ===============================================================

            # Visualization Pipeline
            pipeline.visualization.vis(scenario_index=sample.scenario_index,
                                       scenario=pipeline.scenario,
                                       renderer=pipeline.renderer,
                                       stickers=pipeline.stickers,
                                       smoke=pipeline.smoke)
        except KeyboardInterrupt:
            print("Stop Attack Manually!")
            logger.close_logger()
        logger.close_logger()
        step += 1


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
