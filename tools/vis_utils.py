import cv2
import numpy as np
from comet_ml import Experiment
from matplotlib import pyplot as plt

from smoke.obstacle import Obstacle


def draw_3d_boxes(image, obstacles, color_map=None):
    assert isinstance(image, np.ndarray)
    image = image.copy()
    if image.dtype == np.float64 or image.dtype == np.float32:
        image = image * 255.0
        image = image.astype(np.uint8)
    for obstacle in obstacles:
        image = draw_3d_box(image, obstacle, color_map)
    return image


def draw_3d_box(image, obstacle, color_map=None):
    """
    绘制单个障碍物的3D Box.\n
    :param image: np.uint8 type np.ndarray of image.
    :param obstacle: Obstacle object.
    :param color_map: color map.
    :return: np.ndarray
    """
    if color_map is None:
        color_map = {0: (255, 167, 38), 1: (38, 198, 218), 2: (156, 204, 101)}
    # 2*8 -> 8*2 8 points with x-y coordinates
    corners = np.array(obstacle.box3d).transpose((1, 0))
    class_name = Obstacle.type_map[obstacle.type.item()]
    score = obstacle.score
    color = color_map[obstacle.type.item()]

    # ====================== draw 3d box ======================
    # 车头看向车位
    # 车头所在平面，左上角，顺时针，点为1、4、3、2
    # 车尾所在平面，左上角，顺时针，点为0、5、6、7
    face_idx = [[5, 4, 3, 6],
                [1, 2, 3, 4],
                [1, 0, 7, 2],
                [0, 5, 6, 7]]

    for ind_f in range(3, -1, -1):
        f = face_idx[ind_f]
        for j in range(4):
            cv2.line(image, (int(corners[f[j], 0]), int(corners[f[j], 1])),
                     (int(corners[f[(j + 1) % 4], 0]), int(corners[f[(j + 1) % 4], 1])),
                     color, 2, lineType=cv2.LINE_AA)
        # 绘制斜十字线，车身左侧面（驾驶员所在）
        if ind_f == 0:
            # 车尾上到车头下
            cv2.line(image, (int(corners[f[0], 0]), int(corners[f[0], 1])),
                     (int(corners[f[2], 0]), int(corners[f[2], 1])),
                     color, 1, lineType=cv2.LINE_AA)
            # 车位下到车头上
            cv2.line(image, (int(corners[f[1], 0]), int(corners[f[1], 1])),
                     (int(corners[f[3], 0]), int(corners[f[3], 1])),
                     color, 1, lineType=cv2.LINE_AA)
    # =========================================================

    # ================ draw label(class, score) ===============
    # TODO
    # =========================================================
    return image


def draw_2d_boxes(image, obstacles, color_map=None, from_3d=True):
    assert isinstance(image, np.ndarray)
    image = image.copy()
    if image.dtype == np.float64 or image.dtype == np.float32:
        image = image * 255.0
        image = image.astype(np.uint8)
    for obstacle in obstacles:
        image = draw_2d_box(image, obstacle, color_map, from_3d)
    return image


def draw_2d_box(image, obstacle, color_map=None, from_3d=False):
    """
    绘制单个障碍物的2D Box.\n
    :param image: np.uint8 type np.ndarray of image.
    :param obstacle: Obstacle object.
    :param color_map: color map.
    :param from_3d: whether 2D Box is calculated by 3D Box.
    :return: np.ndarray
    """
    if color_map is None:
        color_map = {0: (255, 167, 38), 1: (38, 198, 218), 2: (156, 204, 101)}
    if from_3d:
        # x_min, y_min, x_max, y_max
        box_2d = [int(min(obstacle.box3d[0])), int(min(obstacle.box3d[1])), int(max(obstacle.box3d[0])),
                  int(max(obstacle.box3d[1]))]
    else:
        box_2d = list(map(int, obstacle.box2d))
    class_name = Obstacle.type_map[obstacle.type.item()]
    score = obstacle.score
    color = color_map[obstacle.type.item()]
    label = class_name + ": " + "%.2f" % score

    # ====================== draw 2d box ======================
    cv2.rectangle(image, (box_2d[0], box_2d[1]), (box_2d[2], box_2d[3]), color=color, thickness=2)
    # =========================================================

    # ================ draw label(class, score) ===============
    # 设置字体格式及大小
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_size = cv2.getTextSize(label, font, 1, 2)
    text_origin = np.array([box_2d[0], box_2d[1] - label_size[0][1]])
    cv2.rectangle(image, tuple(text_origin), tuple(text_origin + label_size[0]), color=color, thickness=-1)
    cv2.putText(image, label, (box_2d[0], box_2d[1]), font, 1, (255, 255, 255), 2)
    # =========================================================

    return image


def plot_img(image: np.ndarray, title: str, dpi=400):
    assert isinstance(image, np.ndarray)
    image = image.copy()
    plt.figure(dpi=dpi)
    if image.dtype == np.float64:
        image = image.astype(np.float32)
    plt.imshow(image)
    plt.title(title)
    plt.grid("off")
    plt.axis('off')
    plt.show()
    plt.close()


def save_img(image: np.ndarray, path: str, is_BGR=False):
    assert isinstance(image, np.ndarray)
    image = image.copy()
    if image.dtype == np.float64 or image.dtype == np.float32:
        image = image * 255.0
        image = image.astype(np.uint8)
    if is_BGR:
        cv2.imwrite(path, image, [cv2.IMWRITE_PNG_COMPRESSION, 4])
    else:
        cv2.imwrite(path, image[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 4])


def log_img(logger: Experiment, image: np.ndarray, img_name: str, step=None):
    assert isinstance(image, np.ndarray)
    image = image.copy()
    if image.dtype == np.float64 or image.dtype == np.float32:
        image = image * 255.0
        image = image.astype(np.uint8)
    logger.log_image(image_data=image, name=img_name, step=step)