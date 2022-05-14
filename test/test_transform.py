import os.path as osp

import cv2
import numpy as np
import torch

from defense.transform import Transform
from tools.vis_utils import plot_img

if __name__ == '__main__':
    project_path = "/home/dingxl/nfs/workspace/smoke-adv/"
    img_path = "data/datasets/waymo/kitti_format/training/image_0"
    img_file = "0002055.png"
    image = cv2.imread(osp.join(project_path, img_path, img_file))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # HWC uint8 -> float32
    image = torch.tensor((image / 255.0).astype(np.float32))

    plot_img(image.numpy(), title="")
    args = dict(type="JpegCompression",
                bit_depth=dict(r_bits=1, g_bits=1, b_bits=1),
                median_blur=dict(kernel_size=21),
                gaussian_blur=dict(kernel_size=29, sigma=0),
                jpeg_compression=dict(quality=10))

    defense = Transform(args)
    purifier_img = defense.forward(image)
    plot_img(defense.visualization["purifier_image"], title="")
    pass
# GB [5, 11, 17, 23, 29, 35, 41, 47]
# MB [5, 9, 13, 17, 21, 25, 29, 33]
# BD [5, 4, 3, 2, 1]
# JC [90, 70, 50, 30, 10]
