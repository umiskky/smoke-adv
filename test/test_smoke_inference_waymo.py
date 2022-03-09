import cv2
import torch

from smoke.dataset.waymo import getKMatrix
from smoke.smoke import Smoke
from smoke.utils import input_utils
from smoke.utils.input_utils import transform_input
from smoke.utils.obstacle import Obstacle
from tools.visualization.vis_detection import draw_3d_boxes, draw_2d_boxes

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    file_idx = "0002045"
    input_path = "../data/datasets/waymo/kitti_format/training/image_0/" + file_idx + ".png"
    image, ori_img_size, _ = transform_input(input_path)
    # image, ori_img_size, _ = getImg("../data/objects/man/test_render_synthesis.jpg")
    smoke = Smoke(device)
    smoke.k = getKMatrix(file_idx)
    print(smoke.k)

    smoke.forward(image, ori_img_size)
    obstacle_list = Obstacle.decode(smoke.box3d_branch, input_utils.getIntrinsicMatrix(False, smoke.k),
                                    ori_img_size)
    print(len(obstacle_list))
    img_draw = cv2.imread(input_path)
    # 遍历每一个预测的bbox_3d
    img_draw = draw_3d_boxes(img_draw, obstacle_list)
    cv2.imwrite("../data/results/attack/" + file_idx + "_3d_raw.png", img_draw)

    img_draw = cv2.imread(input_path)
    # 遍历每一个预测的bbox_3d
    img_draw = draw_2d_boxes(img_draw, obstacle_list)
    cv2.imwrite("../data/results/attack/" + file_idx + "_2d_raw.png", img_draw)
    pass
