import csv
import os

import numpy as np


def getKMatrix(calib_idx, calib_path="../data/datasets/waymo/kitti_format/training/calib/"):
    """
    解析calibration文件获得内参矩阵K
    :param calib_idx: calibration file index.
    :param calib_path: calibration directory path.
    :return: np.ndarray
    """
    # get camera intrinsic matrix K
    with open(os.path.join(calib_path, calib_idx + ".txt"), 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=' ')
        for line, row in enumerate(reader):
            if row[0] == 'P0:':
                K = row[1:]
                K = [float(i) for i in K]
                K = np.array(K, dtype=np.float32).reshape(3, 4)
                K = K[:3, :3]
                return K
