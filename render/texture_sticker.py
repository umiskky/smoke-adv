import os
import os.path as osp

import cv2
import numpy as np
import torch

from tools.color_utils import RgbToHls, HlsToRgb


class TextureSticker:
    def __init__(self, args: dict, textures: torch.Tensor) -> None:
        self._device = args["device"]
        self._texture_hls = self._textures_rgb_to_hls(textures)
        self._sticker_type = args["type"]

        self._patch = None
        self._mask = None
        if "hls" == self._sticker_type:
            # TODO args["texture"] may removed later
            self._patch = self._init_hls_patch(size=args["size"],
                                               position=args["position"],
                                               min_=args["clip_min"],
                                               max_=args["clip_max"],
                                               texture_shape=args["texture"],
                                               device=self._device)
            self._mask = self._init_mask(mask_path=args["mask"],
                                         base_dir=os.getenv("project_path"),
                                         texture_shape=args["texture"],
                                         device=self._device)
        self.visualization = {}

    @property
    def patch(self):
        return self._patch

    @patch.setter
    def patch(self, patch_path):
        project_path = os.getenv("project_path")
        fp = osp.join(project_path, patch_path)
        if osp.exists(fp):
            state = torch.load(fp)
            patch = state.get("patch")
            self._patch = patch.to(self._device)

    def forward(self, mesh, enable_patch_grad=False):
        vis_texture = None
        if "hls" == self._sticker_type:
            # Apply Sticker
            vis_texture = self._apply_hls_sticker(patch=self._patch,
                                                  textures=mesh.textures,
                                                  texture_hls=self._texture_hls.clone(),
                                                  mask=self._mask.clone() if self._mask is not None else None,
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
    def _apply_hls_sticker(patch, textures, texture_hls: torch.Tensor, mask=None, require_grad=False):
        if require_grad and patch.requires_grad is False:
            patch.requires_grad_(True)
        if mask is None:
            texture_hls_patch = texture_hls + patch
        else:
            texture_hls_patch = texture_hls + patch * mask

        texture_hls_patch = texture_hls_patch.squeeze()
        h: torch.Tensor = torch.select(texture_hls_patch, -3, 0)
        l: torch.Tensor = torch.select(texture_hls_patch, -3, 1)
        s: torch.Tensor = torch.select(texture_hls_patch, -3, 2)
        eps = 1e-10
        l_ = torch.clamp(l, min=eps, max=1)
        texture_hls_patch = torch.stack([h, l_, s], dim=-3).unsqueeze(0)

        # HLS -> RGB
        rgb = HlsToRgb()
        texture_rgb_patch = rgb(texture_hls_patch)
        texture_rgb_patch = texture_rgb_patch.permute(0, 2, 3, 1)
        # update texture
        setattr(textures, "_maps_padded", texture_rgb_patch)
        return texture_rgb_patch

    @staticmethod
    def _init_mask(mask_path: dict, base_dir: str, texture_shape, device="cpu"):
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
    def _init_hls_patch(size, position, min_, max_, texture_shape, device="cpu"):
        """Generate aligned patch in channel l perturb"""
        patch = TextureSticker._generate_uniform_tensor(size,
                                                        min_=min_,
                                                        max_=max_,
                                                        device=device)
        # Align Patch
        x_l, y_l = position
        patch_align = torch.zeros(texture_shape, device=device)
        patch_align[:, 1, y_l: y_l + patch.shape[1], x_l: x_l + patch.shape[2]] = patch
        return patch_align

    @staticmethod
    def _generate_uniform_tensor(size, min_: float, max_: float, device="cpu"):
        """Generate uniform from min to max"""
        if isinstance(size, tuple) or isinstance(size, list):
            sticker: torch.Tensor = torch.rand((1, size[0], size[1]), device=device)
            sticker = torch.add(torch.mul(sticker, max_ - min_), min_)

        else:
            sticker: torch.Tensor = torch.rand((1, size, size), device=device)
            sticker = torch.add(torch.mul(sticker, max_ - min_), min_)
        return sticker

    @staticmethod
    def _textures_rgb_to_hls(textures: torch.Tensor) -> torch.Tensor:
        """Convert texture's channel from rgb to hls"""
        # [B, H, W, C] -> [B, C, H, W]
        texture_tensor = textures.permute(0, 3, 1, 2)
        # RGB -> HLS
        hls = RgbToHls()
        texture_tensor_hls = hls(texture_tensor)
        return texture_tensor_hls
