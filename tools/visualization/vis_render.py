import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt


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
        cv2.imwrite(path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    else:
        cv2.imwrite(path, image[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 0])


@DeprecationWarning
def vis_texture(texture, fig_size=5):
    with torch.no_grad():
        plt.figure(figsize=(fig_size, fig_size))
        plt.imshow(texture.squeeze().cpu().numpy())
        plt.grid("off")
        plt.axis('off')
        plt.show()


@DeprecationWarning
def vis_sticker(sticker, fig_size=5):
    with torch.no_grad():
        plt.figure(figsize=(fig_size, fig_size))
        plt.imshow(sticker.clone().cpu().numpy())
        plt.grid("off")
        plt.axis('off')
        plt.show()


@DeprecationWarning
def vis_img(img, fig_size=(5, 5)):
    with torch.no_grad():
        if isinstance(img, torch.Tensor):
            img = img.clone().cpu().numpy()
        if img.max() > 1.0:
            img = img.astype(np.uint8)
        plt.figure(figsize=(fig_size[0], fig_size[1]))
        # plt对于float类型，默认0-1；int类型，默认0-255.
        plt.imshow(img)
        plt.grid("off")
        plt.axis('off')
        plt.show()
