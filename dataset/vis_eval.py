import os.path as osp
import pickle
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

from tools.file_utils import makedirs


def vis(content, title: str, x_label: str, y_label: str, save_path: Optional[str] = None):
    plt.figure(1, dpi=800)
    plt.subplot()
    plt.plot(content)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if save_path:
        plt.savefig(save_path)
    plt.close()
    # plt.show()


def vis2(content, title: str, labels: list, x_label: str, y_label: str, save_path: Optional[str] = None):
    plt.figure(1, dpi=800)
    plt.subplot()
    plt.plot(content[:, 0], color='deepskyblue', label=labels[0])
    plt.plot(content[:, 1], color='coral', label=labels[1])
    plt.plot(content[:, 2], color='mediumseagreen', label=labels[2])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.close()
    # plt.show()


if __name__ == '__main__':
    # Saving Metrics
    fd = osp.join("/home/dingxl/workspace/smoke-adv/", "data/results/eval/dataset_train")
    fp = osp.join(fd, "metrics_eval_dataset.pickle")
    delta_table = PrettyTable()
    delta_table.field_names = ["angle", "location_x", "location_y", "location_z",
                               "dimension_x", "dimension_y", "dimension_z", "iou_2d",
                               "car", "cyclist", "walker", "score"]
    with open(fp, 'rb') as fpb:
        metrics_dict: dict = pickle.load(fpb)
    for angle, metrics in metrics_dict.items():
        # init
        location_list = []
        dimension_list = []
        iou_2d = []
        type_list = []
        car = 0
        cyclist = 0
        walker = 0
        score = []
        # process
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
            "%.3f" % np_score.mean()
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
    print(delta_table)
