import os
import os.path as osp

import cv2
import numpy as np
import torch

from tools.color_utils import RgbToHls, HlsToRgb


class TextureSticker:
    def __init__(self, args: dict, textures: torch.Tensor) -> None:
        self._device = args["device"]
        self._ori_texture_hls = self._textures_rgb_to_hls(textures)
        self._sticker_type = args["type"]

        self._adv_texture_hls = None
        self._mask = None
        if "hls" == self._sticker_type:
            self._adv_texture_hls = self._ori_texture_hls.clone()
            self._mask = self._init_mask(mask_path=args["mask"],
                                         base_dir=os.getenv("project_path"),
                                         device=self._device)
        self.visualization = {}

    @property
    def mask(self):
        return self._mask.clone()

    @property
    def adv_texture_hls(self):
        return self._adv_texture_hls

    @adv_texture_hls.setter
    def adv_texture_hls(self, adv_texture_path):
        """Using for loading adv texture from file."""
        project_path = os.getenv("project_path")
        fp = osp.join(project_path, adv_texture_path)
        if osp.exists(fp):
            state = torch.load(fp)
            adv_texture = state.get("adv_texture")
            self._adv_texture_hls = adv_texture.to(self._device)

    def apply_gauss_perturb(self, clip_min=-0.1, clip_max=0.1):
        """Apply normal perturb to ori texture."""
        ori_texture_hls = self._ori_texture_hls.clone()
        # X = z * sigma + mu
        random_negative = torch.add(torch.mul(torch.randn(ori_texture_hls.shape, device=ori_texture_hls.device),
                                              abs(clip_min) / 3), clip_min)
        random_negative = torch.clamp(random_negative, min=-1.0, max=1.0)
        random_positive = torch.add(torch.mul(torch.randn(ori_texture_hls.shape, device=ori_texture_hls.device),
                                              abs(clip_max) / 3), clip_max)
        random_positive = torch.clamp(random_positive, min=-1.0, max=1.0)
        random_positive_mask = torch.ge(torch.randn(ori_texture_hls.shape, device=ori_texture_hls.device), 0)
        random_negative_mask = torch.lt(torch.randn(ori_texture_hls.shape, device=ori_texture_hls.device), 0)
        random = random_positive_mask * random_positive + random_negative_mask * random_negative
        adv_texture_hls = ori_texture_hls + self._mask * random
        self._adv_texture_hls = adv_texture_hls

    @property
    def ori_texture_hls(self):
        return self._ori_texture_hls.clone()

    def forward(self, mesh, enable_patch_grad=False):
        vis_texture = None
        if "hls" == self._sticker_type:
            # Apply Sticker
            vis_texture = self._apply_hls_sticker(textures=mesh.textures,
                                                  texture_hls=self._adv_texture_hls,
                                                  require_grad=enable_patch_grad)
        # ======================================= Visualization =======================================
        with torch.no_grad():
            if vis_texture is not None:
                if vis_texture.requires_grad:
                    vis_texture = vis_texture.detach().clone().cpu()
                else:
                    vis_texture = vis_texture.clone().cpu()
                # 0~1.0 RGB HWC
                self.visualization["texture"] = vis_texture.squeeze().numpy()
                self.visualization["texture_perturb"] = vis_texture.squeeze()
        # =============================================================================================
        return mesh

    @staticmethod
    def _apply_hls_sticker(textures, texture_hls: torch.Tensor, require_grad=False):
        if require_grad and texture_hls.requires_grad is False:
            texture_hls.requires_grad_(True)

        texture_hls_patch = texture_hls

        texture_hls_patch = texture_hls_patch.squeeze()
        h: torch.Tensor = torch.select(texture_hls_patch, -3, 0)
        l: torch.Tensor = torch.select(texture_hls_patch, -3, 1)
        s: torch.Tensor = torch.select(texture_hls_patch, -3, 2)
        eps = 1e-10
        l_ = torch.clamp(l, min=eps, max=1 - eps)
        texture_hls_patch = torch.stack([h, l_, s], dim=-3).unsqueeze(0)

        # HLS -> RGB
        rgb = HlsToRgb()
        texture_rgb_patch = rgb(texture_hls_patch)
        texture_rgb_patch = texture_rgb_patch.permute(0, 2, 3, 1)
        # update texture
        setattr(textures, "_maps_padded", texture_rgb_patch)
        return texture_rgb_patch

    @staticmethod
    def _init_mask(mask_path: dict, base_dir: str, device="cpu"):
        mask = None
        if len(mask_path) > 0:
            for mkp in mask_path.values():
                mkp_abs = osp.join(base_dir, mkp)
                mk_img_bgr = cv2.imread(mkp_abs)
                mk_img_rgb = cv2.cvtColor(mk_img_bgr, cv2.COLOR_BGR2RGB)
                mk_img_norm = mk_img_rgb.astype(np.float32) / 255.0
                mk_tensor = torch.tensor(mk_img_norm, device=device).permute(2, 0, 1).unsqueeze(0)
                if mask is None:
                    mask = mk_tensor
                else:
                    mask = mask + mk_tensor
        # merge rgb channel & clamp to 0 or 1.
        mask_l = torch.sum(mask, dim=1).ge(1.5)
        mask_h = mask_s = torch.zeros_like(mask_l)
        mask = torch.cat([mask_h, mask_l, mask_s], dim=0)[None, ...]
        return mask

    @staticmethod
    def _textures_rgb_to_hls(textures: torch.Tensor) -> torch.Tensor:
        """Convert texture's channel from rgb to hls"""
        # [B, H, W, C] -> [B, C, H, W]
        texture_tensor = textures.permute(0, 3, 1, 2)
        # RGB -> HLS
        hls = RgbToHls()
        texture_tensor_hls = hls(texture_tensor)
        return texture_tensor_hls
