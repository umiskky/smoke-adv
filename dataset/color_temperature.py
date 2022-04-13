import numpy as np
import torch

from dataset.ycbcr_transformer import RgbToYcbcr, YcbcrToRgb


class ColorTemperature:
    def __init__(self, phi) -> None:
        self._phi = phi

    def calculate_temperature(self, image_rgb: torch.Tensor) -> list:
        temperature = []
        rgb_to_ycbcr = RgbToYcbcr()
        image_ycbcr = rgb_to_ycbcr(image_rgb)
        image_ycbcr = image_ycbcr.squeeze()
        y: torch.Tensor = image_ycbcr[..., 0, :, :]
        cb: torch.Tensor = image_ycbcr[..., 1, :, :]
        cr: torch.Tensor = image_ycbcr[..., 2, :, :]
        cb_abs = torch.abs(cb)
        cr_abs = torch.abs(cr)
        mask = torch.ge(input=y - cb_abs - cr_abs, other=self._phi)
        y_masked = torch.masked_select(y, mask)
        cb_masked = torch.masked_select(cb, mask)
        cr_masked = torch.masked_select(cr, mask)
        temperature_ycbcr = torch.tensor([y_masked.mean(), cb_masked.mean(), cr_masked.mean()])[..., None, None]
        ycbcr_to_rgb = YcbcrToRgb()
        # CHW 0.0~1.0
        temperature_ycbcr_rgb = ycbcr_to_rgb(temperature_ycbcr)
        # 0~255
        temperature = torch.clamp((temperature_ycbcr_rgb * 255.0).int(), min=0, max=255)
        temperature = temperature.squeeze().numpy().tolist()
        return temperature

    @staticmethod
    def transform_image(image: np.ndarray) -> torch.Tensor:
        """image: np.ndarray 0~255 HWC RGB -> torch.Tensor 0.0~1.0 CHW RGB """
        image_ = image.copy() / 255.0
        device = "cuda:0"
        image_ = torch.tensor(image_, device=device)
        # avoid numerical problem
        image_ = torch.clamp(image_, min=0.0, max=1.0)
        # HWC -> CHW
        image_ = torch.permute(image_, [2, 0, 1])
        return image_
