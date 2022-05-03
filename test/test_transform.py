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
                bit_depth=dict(r_bits=8, g_bits=8, b_bits=8),
                median_blur=dict(kernel_size=11),
                gaussian_blur=dict(kernel_size=3, sigma=0),
                jpeg_compression=dict(quality_lower=0.0, quality_upper=10.0))

    defense = Transform(args)
    purifier_img = defense.forward(image)
    plot_img(defense.visualization["purifier_image"], title="")
    pass
