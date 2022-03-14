import torch
import torch.nn as nn

from render.utils.color_utils import RgbToHls, HlsToRgb


class TextureSticker(nn.Module):
    def __init__(self, args: dict, global_args: dict) -> None:
        super().__init__()
        self.device = torch.device(args["device"])
        self.sticker = None
        self.patch = None
        self.is_attack = global_args["attack"]
        self.sticker_type = args["type"]
        self.position = args["position"]
        self.intensity = args["intensity"]
        self.texture_shape = args["texture"]
        self.visualization = {}

        if "hls" == self.sticker_type:
            self.generate_uniform__11_sticker(size=args["size"],  device=self.device)

    def forward(self, mesh):
        if "hls" == self.sticker_type:
            return self.apply_hls_sticker(mesh=mesh,
                                          intensity=self.intensity)
        return mesh

    def apply_hls_sticker(self, mesh, intensity=0.5):
        assert self.sticker is not None
        with torch.no_grad():
            texture = mesh.textures
            # [B, H, W, C]
            texture_tensor = getattr(texture, "_maps_padded")
            # texture_tensor.requires_grad_(False)
            # [B, C, H, W]
            texture_tensor = texture_tensor.permute(0, 3, 1, 2)
            # texture_tensor.requires_grad_(False)
            # RGB -> HLS
            hls = RgbToHls()
            texture_tensor_hls = hls(texture_tensor)

        self.patch.requires_grad_(True)
        texture_tensor_hls_patch = texture_tensor_hls + intensity * self.patch

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

    def generate_uniform__11_sticker(self, size, device=torch.device("cpu")):
        if isinstance(size, tuple) or isinstance(size, list):
            self.sticker: torch.Tensor = (torch.rand((1, size[0], size[1]), device=device) - 0.5) * 2.0
        else:
            self.sticker: torch.Tensor = (torch.rand((1, size, size), device=device) - 0.5) * 2.0
        # Add Patch
        x_l, y_l = self.position
        patch = torch.zeros(self.texture_shape, device=device)
        patch[:, 1, y_l: y_l + self.sticker.shape[1], x_l: x_l + self.sticker.shape[2]] = self.sticker
        self.patch = patch
        if self.is_attack:
            self.patch.requires_grad_(True)

    @staticmethod
    def generate_uniform_01_sticker(size, device=torch.device("cpu")):
        """
        Generate sticker .\n
        :param size: tuple(H, W) OR square size.
        :param device: device.
        :return: torch.Tensor
        """
        if isinstance(size, tuple):
            return torch.rand((size[0], size[1], 3), device=device)
        return torch.rand((size, size, 3), device=device)

    @DeprecationWarning
    def apply_rgb_sticker(self, mesh, position, intensity=0.01, ops="add"):
        """
        Apply new texture to mesh of object.\n
        :param mesh: mesh object.
        :param position: start position of sticker(x->W, y->H).
        :param intensity: add operation intensity.
        :param ops: apply operations. ("add" | "mul").
        :return: None
        """
        assert self.sticker is not None
        texture = mesh.textures
        texture_tensor = getattr(texture, "_maps_padded")

        patch = torch.zeros_like(texture, device=self.device)
        x_l, y_l = position
        patch[:, y_l: y_l + self.sticker.shape[0], x_l: x_l + self.sticker.shape[1], :] = self.sticker

        if "add" == ops:
            texture_tensor[:] = intensity * patch + texture_tensor
        if "mul" == ops:
            mask = torch.ones_like(texture, device=self.device)
            mask[:, y_l: y_l + self.sticker.shape[0], x_l: x_l + self.sticker.shape[1]] = 0
            texture_tensor[:] = mask * patch + texture_tensor
        # avoid color value invalid
        texture_tensor[:] = torch.clamp(texture_tensor, min=0.0, max=1.0)
