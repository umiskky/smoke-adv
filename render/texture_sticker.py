import numpy as np
import torch
import torch.nn as nn
import os.path as osp
import cv2

from render.utils.color_utils import RgbToHls, HlsToRgb


class TextureSticker(nn.Module):
    def __init__(self, args: dict, global_args: dict) -> None:
        super().__init__()
        self.device = torch.device(args["device"])
        self.patch = None
        self.mask = None
        self.is_attack = global_args["attack"]
        self.sticker_type = args["type"]
        self.intensity = args["intensity"]
        self.visualization = {}

        if "hls" == self.sticker_type:
            self.patch = TextureSticker.init_hls_patch(size=args["size"],
                                                       position=args["position"],
                                                       min_=args["clip_min"],
                                                       max_=args["clip_max"],
                                                       texture_shape=args["texture"],
                                                       require_grad=self.is_attack,
                                                       device=self.device)
            self.mask = TextureSticker.init_mask(mask_path=args["mask"],
                                                 base_dir=global_args["project_path"],
                                                 texture_shape=args["texture"],
                                                 device=self.device)

    def forward(self, mesh):
        if "hls" == self.sticker_type:
            with torch.no_grad():
                texture = mesh.textures
                # [B, H, W, C]
                texture_tensor = getattr(texture, "_maps_padded")
                # [B, C, H, W]
                texture_tensor = texture_tensor.permute(0, 3, 1, 2)
                # RGB -> HLS
                hls = RgbToHls()
                texture_tensor_hls = hls(texture_tensor)
                mask = self.mask.clone()
            # Apply Sticker
            texture_tensor_rgb_patch = self.apply_hls_sticker(patch=self.patch,
                                                              texture=texture,
                                                              texture_tensor_hls=texture_tensor_hls,
                                                              mask=mask,
                                                              intensity=self.intensity,
                                                              require_grad=self.is_attack)
            # ======================================= Visualization =======================================
            with torch.no_grad():
                if texture_tensor_rgb_patch.requires_grad:
                    vis_texture = texture_tensor_rgb_patch.detach().clone().cpu()
                else:
                    vis_texture = texture_tensor_rgb_patch.clone().cpu()
                # 0~1.0 RGB HWC
                self.visualization["texture"] = vis_texture.squeeze().numpy()
            # =============================================================================================
        return mesh

    @staticmethod
    def apply_hls_sticker(patch, texture, texture_tensor_hls: torch.Tensor, mask, intensity=0.5, require_grad=False):
        assert patch is not None
        if require_grad and patch.requires_grad is False:
            patch.requires_grad_(True)
        if mask is None:
            texture_tensor_hls_patch = texture_tensor_hls + intensity * patch
        else:
            texture_tensor_hls_patch = texture_tensor_hls + intensity * patch * mask

        texture_tensor_hls_patch = texture_tensor_hls_patch.squeeze()
        h: torch.Tensor = torch.select(texture_tensor_hls_patch, -3, 0)
        l: torch.Tensor = torch.select(texture_tensor_hls_patch, -3, 1)
        s: torch.Tensor = torch.select(texture_tensor_hls_patch, -3, 2)
        l_ = torch.clamp(l, min=0, max=1)
        texture_tensor_hls_patch = torch.stack([h, l_, s], dim=-3).unsqueeze(0)

        # HLS -> RGB
        rgb = HlsToRgb()
        texture_tensor_rgb_patch = rgb(texture_tensor_hls_patch)
        texture_tensor_rgb_patch = texture_tensor_rgb_patch.permute(0, 2, 3, 1)
        # update texture
        setattr(texture, "_maps_padded", texture_tensor_rgb_patch)
        return texture_tensor_rgb_patch

    @staticmethod
    def init_mask(mask_path: dict, base_dir: str, texture_shape, device=torch.device("cpu")):
        if len(mask_path) > 0:
            mask = torch.zeros(size=texture_shape, device=device)
            for mkp in mask_path.values():
                mkp_abs = osp.join(base_dir, mkp)
                mk_img_bgr = cv2.imread(mkp_abs)
                mk_img_rgb = cv2.cvtColor(mk_img_bgr, cv2.COLOR_BGR2RGB)
                mk_img_norm = mk_img_rgb.astype(np.float32) / 255.0
                mk_tensor = torch.tensor(mk_img_norm, device=device).permute(2, 0, 1).unsqueeze(0)
                mask = mask + mk_tensor
            return mask
        return None

    @staticmethod
    def init_hls_patch(size, position, min_, max_, texture_shape, require_grad=False, device=torch.device("cpu")):
        sticker = TextureSticker.generate_uniform_tensor(size,
                                                         min_=min_,
                                                         max_=max_,
                                                         device=device)
        # Add Patch
        x_l, y_l = position
        patch = torch.zeros(texture_shape, device=device)
        patch[:, 1, y_l: y_l + sticker.shape[1], x_l: x_l + sticker.shape[2]] = sticker
        if require_grad:
            patch.requires_grad_(True)
        return patch

    @staticmethod
    def generate_uniform_tensor(size, min_: float, max_: float, device=torch.device("cpu")):
        """Generate uniform from min to max"""
        if isinstance(size, tuple) or isinstance(size, list):
            sticker: torch.Tensor = torch.rand((1, size[0], size[1]), device=device)
            # sticker = torch.mul(sticker.add(-0.5), 2.0)
            sticker = torch.add(torch.mul(sticker, max_-min_), min_)

        else:
            sticker: torch.Tensor = torch.rand((1, size, size), device=device)
            # sticker = torch.mul(sticker.add(-0.5), 2.0)
            sticker = torch.add(torch.mul(sticker, max_ - min_), min_)
        return sticker
