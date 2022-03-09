import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.renderer import look_at_view_transform, PerspectiveCameras, AmbientLights, RasterizationSettings, \
    BlendParams, MeshRenderer, MeshRasterizer, HardPhongShader, PointLights


class Renderer(nn.Module):
    def __init__(self, args: dict) -> None:
        super().__init__()
        self.device = torch.device(args["device"])
        self.camera = None
        self.light = None
        self.renderer = None

        self.background_color = tuple(args["render"]["background_color"])
        self.scenario_size = args["scenario_size"]

        self.set_camera(height=args["camera"]["height"],
                        K=args["camera"]["K"],
                        img_size=args["scenario_size"])
        self.set_light(light_args=args["light"],
                       light_type=args["light"]["type"])
        self.set_render(quality_rate=args["render"]["quality_rate"],
                        img_size=self.scenario_size,
                        background_color=self.background_color)

        self.visualization = {}

    def forward(self, mesh, scenario):
        # preprocessing ...
        scenario_tensor = self.get_normalized_img_tensor(scenario, self.device)
        # HWC -> RGB -> 0~1.0 torch.Tensor
        synthesis_normalized_img = self.render(mesh, scenario_tensor)
        # 0~255.0
        synthesis_img = synthesis_normalized_img * 255.0
        return synthesis_img

    def set_camera(self, height, K, img_size):
        """
        Set camera position and parameters.\n
        :param height: height of the camera to the ground.
        :param K: 3x3 camera intrinsic matrix.
        :param img_size: tuple(h, w) used for ndc coordinate.
        :return: None
        """
        # World Coordinate   Camera Coordinate
        #             back to front
        #        ^ y                ^ y
        #        |                  |
        #      z ⊙--> x      x <-- ⊕ z
        # camera position in world: (0, h, 0). h is the height of the camera to the ground.
        R, T = look_at_view_transform(eye=((0, height, 0),), at=((0, height, -1),), device=self.device)
        self.camera = PerspectiveCameras(focal_length=((K[0, 0], K[1, 1]),),
                                         principal_point=((K[0, 2], K[1, 2]),),
                                         in_ndc=False,
                                         image_size=(img_size,),
                                         R=R,
                                         T=T,
                                         device=self.device)

    def set_light(self, light_args: dict, light_type="point"):
        """
        TODO light settings.
        :return:
        """
        if "ambient" == light_type:
            self.light = AmbientLights(device=self.device)
        if "point" == light_type:
            self.light = PointLights(ambient_color=(tuple(light_args["ambient_color"]),),
                                     diffuse_color=(tuple(light_args["diffuse_color"]),),
                                     specular_color=(tuple(light_args["specular_color"]),),
                                     location=(tuple(light_args["location"]),),
                                     device=self.device)

    def set_render(self, quality_rate, img_size, background_color=(0.0, 0.0, 0.0)):
        """
        Set renderer parameters.\n
        :param quality_rate: quality rate, used for render quality.
        :param img_size: output image size.
        :param background_color: render background.
        :return: None
        """
        assert self.camera is not None
        assert self.light is not None
        # Rasterization Setting
        raster_settings = RasterizationSettings(
            image_size=(img_size[0] * quality_rate, img_size[1] * quality_rate),
        )
        # Blend Setting
        blendParams = BlendParams(
            background_color=background_color,
        )
        # Renderer
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.camera,
                raster_settings=raster_settings
            ),
            shader=HardPhongShader(
                device=self.device,
                cameras=self.camera,
                lights=self.light,
                blend_params=blendParams
            )
        )

    def render(self, mesh, target) -> torch.Tensor:
        assert self.renderer is not None
        # render mesh in the background
        mesh_in_back = self.renderer(mesh)
        # TODO
        # C H W
        mesh_in_back = F.adaptive_avg_pool2d(mesh_in_back[0, ..., :3].permute(2, 0, 1).unsqueeze(0),
                                             (self.scenario_size[0], self.scenario_size[1]))
        # H W C
        mesh_in_back = mesh_in_back[0, :].permute(1, 2, 0)
        synthesis_img = self.merge_render_target(mesh_in_back, target)
        # if vis:
        #     vis_img(mesh_in_back, (mesh_in_back.shape[1] / 100, mesh_in_back.shape[0] / 100))
        # if save:
        #     save_img(mesh_in_back, save_path, is_normalized=True)

        # ======================================= Visualization =======================================
        if mesh_in_back.requires_grad:
            vis_mesh_in_back = mesh_in_back.detach().clone().cpu().numpy()
        else:
            vis_mesh_in_back = mesh_in_back.clone().cpu().numpy()
        if synthesis_img.requires_grad:
            vis_synthesis_img = synthesis_img.detach().clone().cpu().numpy()
        else:
            vis_synthesis_img = synthesis_img.clone().cpu().numpy()
        self.visualization["render_bg"] = vis_mesh_in_back
        self.visualization["render_scenario"] = vis_synthesis_img
        # =============================================================================================

        return synthesis_img

    def merge_render_target(self, mesh_in_back, target) -> torch.Tensor:
        synthesis_img = None
        # TODO 暂时背景设计为黑色，方便提取mask
        if self.background_color == (0.0, 0.0, 0.0):
            mask_indexes = torch.nonzero(mesh_in_back.sum(2))
            mask_indexes = mask_indexes[:, 0], mask_indexes[:, 1]
            mask = torch.ones(mesh_in_back.shape, device=self.device)
            mask = mask.index_put(mask_indexes, torch.tensor([0.0, 0.0, 0.0], device=self.device))
            synthesis_img = target * mask + mesh_in_back
        return synthesis_img

    @staticmethod
    def get_normalized_img_tensor(image: np.ndarray, device=torch.device("cpu")):
        """
        Get normalized image tensor in RGB type.\n
        :param image: HWC -> RGB -> 0~255 np.ndarray.
        :param device: tensor device.
        :return: Tensor(H, W, 3) of image.
        """
        img = image / 255.0
        return torch.tensor(img, device=device)