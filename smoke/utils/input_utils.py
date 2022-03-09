import albumentations as A
import cv2
import numpy as np
import torch
from torchvision.transforms import transforms


def getIntrinsicMatrix(is_inverse=False, k=None, device=torch.device("cpu")):
    """
    相机内参矩阵.\n
    :param is_inverse: 是否求逆
    :param k: 原始内参矩阵
    :param device: device
    :return: Tensor
    """
    if k is None:
        k = np.array([[[2055.56, 0, 939.658], [0, 2055.56, 641.072], [0, 0, 1]]], np.float32)
    if type(k) is not np.ndarray:
        k = np.array(k, np.float32)
    if len(k.shape) == 2:
        k = k[None, :, :]
    if is_inverse:
        k = np.linalg.inv(k)
    return torch.tensor(k, device=device)


def get_ratio(ori_img_size, output_size, down_ratio=(4, 4), device=torch.device("cpu")):
    """
    获取smoke模型输入ratio(x->W, y->H).\n
    :param ori_img_size: 原始图片大小
    :param output_size: 期望输出大小
    :param down_ratio: 默认(4, 4)
    :param device: device
    :return: Tensor
    """
    return torch.tensor(np.array([[down_ratio[1] * ori_img_size[1] / output_size[1],
                                   down_ratio[0] * ori_img_size[0] / output_size[0]]], np.float32),
                        device=device)


def transform_input(img_path, device=torch.device("cpu")) -> (torch.Tensor, list, list):
    """
    进行模型输入变换.\n
    :param img_path: image file path
    :param device: cpu or cuda
    :return: torch.Tensor
    """
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ori_img_size = image.shape
    transform = A.Compose([
        A.Resize(height=640, width=960)
    ])
    transformed = transform(image=image)

    output_size = transformed['image'].shape
    image = np.true_divide(transformed['image'], np.array([58.395, 57.12, 57.375]))
    image = np.array(image, np.float32)
    image = image.transpose((2, 0, 1))[None, :, :, :]
    image = torch.tensor(image, device=device)
    return image, ori_img_size, output_size


def transform_input_tensor(image, device=torch.device("cpu")) -> (torch.Tensor, list, list):
    """
    进行模型输入变换.\n
    :param image: Tensor of normalized image file [0~1]
    :param device: cpu or cuda
    :return: torch.Tensor
    """
    # (H*W*C) 0~1.0
    ori_img_size = image.shape
    # (H*W*C) 0~255.0
    image[:] = image * 255.0
    # (C*H*W) 0~255.0
    image = image.permute((2, 0, 1))
    transform = transforms.Compose([
        transforms.Resize(size=(640, 960)),
        transforms.Normalize(mean=(0, 0, 0), std=(58.395, 57.12, 57.375))
    ])
    image = transform(image)
    output_size = (image.shape[1], image.shape[2], image.shape[0])
    image = image.unsqueeze(0).float()
    return image, ori_img_size, output_size
