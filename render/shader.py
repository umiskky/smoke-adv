from typing import Optional, Union

import torch
from pytorch3d.common import Device
from pytorch3d.renderer import SoftPhongShader, TensorProperties, Materials, BlendParams, phong_shading
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.structures import Meshes


class Shader(SoftPhongShader):
    def __init__(self, device: Device = "cpu",
                 cameras: Optional[TensorProperties] = None,
                 lights: Optional[TensorProperties] = None,
                 materials: Optional[Materials] = None,
                 blend_params: Optional[BlendParams] = None) -> None:
        super().__init__(device, cameras, lights, materials, blend_params)

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs):
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of SoftPhongShader"
            raise ValueError(msg)

        texels = meshes.sample_textures(fragments)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        blend_params = kwargs.get("blend_params", self.blend_params)
        background = kwargs.get("background", blend_params.background_color)
        colors = phong_shading(
            meshes=meshes,
            fragments=fragments,
            texels=texels,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )
        znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))
        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 500.0))
        images, mask = self.softmax_rgb_blend(
            colors, background, fragments, blend_params, znear=znear, zfar=zfar
        )
        return images, mask

    @staticmethod
    def softmax_rgb_blend(
            colors: torch.Tensor,
            background: torch.Tensor,
            fragments,
            blend_params: BlendParams,
            znear: Union[float, torch.Tensor] = 1.0,
            zfar: Union[float, torch.Tensor] = 100.0,
    ):
        """
        [0] Shichen Liu et al, 'Soft Rasterizer: A Differentiable Renderer for
        Image-based 3D Reasoning'
        [1] Johnson et al, 'Accelerating 3D deep learning with PyTorch3D'. SIGGRAPH Asia 2020 Courses.
        """

        N, H, W, K = fragments.pix_to_face.shape
        device = fragments.pix_to_face.device
        pixel_colors = torch.ones((N, H, W, 4), dtype=colors.dtype, device=colors.device)
        if not isinstance(background, torch.Tensor):
            background_ = torch.tensor(background, dtype=torch.float32, device=device)
        else:
            assert background.shape == torch.Size([N, H, W, 3])
            background_ = background.to(device)

        # Weight for background color
        eps = 1e-10

        # Mask for padded pixels.
        mask = fragments.pix_to_face >= 0

        # fragments.dists [-1, 0] which indicate that this implement do not consider sign indicator.
        prob_map = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask

        # The cumulative product ensures that alpha will be 0.0 if at least 1
        # face fully covers the pixel as for that face, prob will be 1.0.
        # This results in a multiplication by 0.0 because of the (1.0 - prob)
        # term. Therefore 1.0 - alpha will be 1.0.
        alpha = torch.prod((1.0 - prob_map), dim=-1)

        # Weights for each face. Adjust the exponential by the max z to prevent
        # overflow. zbuf shape (N, H, W, K), find max over K.

        # Reshape to be compatible with (N, H, W, K) values in fragments
        if torch.is_tensor(zfar):
            zfar = zfar[:, None, None, None]
        if torch.is_tensor(znear):
            znear = znear[:, None, None, None]

        z_inv = (zfar - fragments.zbuf) / (zfar - znear) * mask
        z_inv_max = torch.max(z_inv, dim=-1).values[..., None].clamp(min=eps)
        weights_num = prob_map * torch.exp((z_inv - z_inv_max) / blend_params.gamma)

        # Also apply exp normalize trick for the background color weight.
        # Clamp to ensure delta is never 0.
        delta = torch.exp((eps - z_inv_max) / blend_params.gamma).clamp(min=eps)

        # Normalize weights.
        # weights_num shape: (N, H, W, K). Sum over K and divide through by the sum.
        denom = weights_num.sum(dim=-1)[..., None] + delta

        # Sum: weights * textures + background color
        weighted_colors = (weights_num[..., None] * colors).sum(dim=-2)
        weighted_background = delta * background_
        pixel_colors[..., :3] = (weighted_colors + weighted_background) / denom
        pixel_colors[..., 3] = 1.0 - alpha
        # using torch.clamp to avoid numerical problem
        return torch.clamp(pixel_colors, min=0.0, max=1.0), mask

