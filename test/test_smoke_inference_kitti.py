import csv
import os

import cv2
import numpy as np
import torch

from smoke.smoke import Smoke
from smoke.utils import input_utils
from smoke.utils.input_utils import transform_input
from smoke.utils.obstacle import Obstacle
from tools.visualization.vis_detection import draw_3d_boxes


def getKMatrix(label_idx):
    # get camera intrinsic matrix K
    with open(os.path.join("../data/datasets/kitti/training/calib/", label_idx + ".txt"), 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=' ')
        for line, row in enumerate(reader):
            if row[0] == 'P2:':
                K = row[1:]
                K = [float(i) for i in K]
                K = np.array(K, dtype=np.float32).reshape(3, 4)
                K = K[:3, :3]
                return K


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    file_idx = "000448"
    input_path = "../data/datasets/kitti/training/image_2/" + file_idx + ".png"
    image, ori_img_size, _ = transform_input("../data/objects/man/test_render_synthesis.jpg")
    smoke = Smoke(device)
    Smoke.confidence_threshold = 0.05
    smoke.k = getKMatrix(file_idx)
    print(smoke.k)
    smoke.forward(image, ori_img_size)

    obstacle_list = Obstacle.decode(smoke.box3d_branch, input_utils.getIntrinsicMatrix(False, smoke.k),
                                    ori_img_size)

    # img_draw = cv2.imread(input_path)
    img_draw = cv2.imread("../data/objects/man/test_render_synthesis.jpg")
    # 遍历每一个预测的bbox_3d
    img_draw = draw_3d_boxes(img_draw, obstacle_list)
    cv2.imwrite("../data/examples/" + file_idx + "_test.png", img_draw)
    pass
