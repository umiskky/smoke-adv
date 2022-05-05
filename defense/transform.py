import copy

import albumentations as A
import numpy as np
import torch

from defense.gaussian_blur import GaussianBlur
from defense.image_compression import JpegCompression
from defense.median_blur import MedianBlur


class Transform:
    def __init__(self, args: dict) -> None:
        self._device = None
        self._type = args["type"]
        self._transform = None
        if "BitDepth" == self._type:
            r_bits = self._to_list(args["bit_depth"]["r_bits"])
            g_bits = self._to_list(args["bit_depth"]["g_bits"])
            b_bits = self._to_list(args["bit_depth"]["b_bits"])
            self._transform = A.Compose([
                A.FromFloat(dtype="uint8", always_apply=True),
                A.Posterize(num_bits=[r_bits, g_bits, b_bits], always_apply=True),
                A.ToFloat(always_apply=True)
            ])
        elif "MedianBlur" == self._type:
            kernel_size = args["median_blur"]["kernel_size"]
            self._transform = A.Compose([
                # the kernel size used in median blur is limited to [3, 5] for float 32 in cv.
                A.FromFloat(dtype="uint8", always_apply=True),
                MedianBlur(kernel_size=kernel_size, always_apply=True),
                A.ToFloat(always_apply=True)
            ])
        elif "GaussianBlur" == self._type:
            kernel_size = args["gaussian_blur"]["kernel_size"]
            sigma = args["gaussian_blur"]["sigma"]
            self._transform = A.Compose([
                GaussianBlur(kernel_size=kernel_size, sigma=sigma, always_apply=True)
            ])
        elif "JpegCompression" == self._type:
            self._transform = A.Compose([
                A.FromFloat(dtype="uint8", always_apply=True),
                JpegCompression(quality=args["jpeg_compression"]["quality"],  always_apply=True),
                A.ToFloat(always_apply=True)
            ])
        self.visualization = {}

    def forward(self, image):
        np_image = None
        # Convert torch.Tensor to np.ndarray
        if isinstance(image, torch.Tensor):
            self._device = image.device
            np_image = image.cpu().numpy()
        elif isinstance(image, np.ndarray):
            np_image = image
        if self._transform is not None:
            purifier_img = self._transform(image=np_image)['image']
            self.visualization["purifier_image"] = copy.deepcopy(purifier_img)
            # Convert to torch.Tensor if the type of input is torch.Tensor.
            if self._device is not None:
                purifier_img = torch.tensor(purifier_img, device=self._device)
                self._device = None
            return purifier_img
        else:
            return image

    @staticmethod
    def _to_list(item):
        if isinstance(item, list):
            return item
        return [item, item]