import numpy as np
import torch
from pytorch3d.renderer import look_at_view_transform, PerspectiveCameras, AmbientLights, RasterizationSettings, \
    BlendParams, MeshRasterizer, PointLights

from pipeline.modules.sample import Sample
from render.mesh_renderer import MeshRendererWithMask
from render.shader import Shader


class Renderer:
    def __init__(self, args: dict) -> None:
        super().__init__()
        self._device = args["device"]
        self._camera_height = args["camera"]["height"]
        self._light_type = args["light"]["type"]
        self._background_color = tuple(args["render"]["background_color"])
        self._quality_rate = args["render"]["quality_rate"]
        self._blend_params = BlendParams(
            background_color=self._background_color,
            sigma=float(args["render"]["sigma"]),
            gamma=float(args["render"]["gamma"]),
        )
        self._render_size = args["render"]["image_shape"]

        # Pseudo 2D&3D Box GT
        self._box_pseudo_gt = {"3d": {"h_offset": float(self._camera_height)}}
        self.visualization = {}

    def forward(self, mesh, scenario, sample: Sample):

        # Init
        synthesis_img = render_in_scenario = render_in_bg = None
        render_size = scenario.shape[:2] if scenario is not None else self._render_size

        # =========================== create new renderer ===========================
        camera = self._init_camera(height=self._camera_height, K=sample.K, img_size=render_size, device=self._device)
        light = self._init_light(light_type=self._light_type, device=self._device,
                                 ambient_color=sample.light_ambient_color,
                                 diffuse_color=sample.light_diffuse_color,
                                 specular_color=sample.light_specular_color,
                                 location=sample.light_location)
        renderer = self._init_renderer(quality_rate=self._quality_rate, img_size=render_size,
                                       blend_params=self._blend_params, camera=camera,
                                       light=light, device=self._device)
        # ===========================================================================

        # ================================== Render =================================
        if scenario is not None:
            # preprocessing ...
            scenario_tensor = self._get_normalized_img_tensor(scenario, self._device)
            # render
            render_result, mask = renderer(mesh, background=scenario_tensor.unsqueeze(0))
            synthesis_normalized_img = render_in_bg = render_in_scenario = render_result[0, ..., :3]
            # save some pseudo gt information
            self._box_pseudo_gt["2d"] = self._eval_2d_pseudo_gt(mask)
            self._box_pseudo_gt["3d"]["location"] = sample.location
            # 0~1.0 float32 HWC
            synthesis_img = synthesis_normalized_img
        # ===========================================================================

        # ============================== Visualization ==============================
        with torch.no_grad():
            if render_in_bg.requires_grad:
                vis_mesh_in_back = render_in_bg.detach().clone().cpu().numpy()
            else:
                vis_mesh_in_back = render_in_bg.clone().cpu().numpy()
            if render_in_scenario is not None:
                if render_in_scenario.requires_grad:
                    vis_synthesis_img = render_in_scenario.detach().clone().cpu().numpy()
                else:
                    vis_synthesis_img = render_in_scenario.clone().cpu().numpy()
            # 0~1.0 RGB HWC float64
            self.visualization["render_bg"] = vis_mesh_in_back
            self.visualization["render_scenario"] = vis_synthesis_img
        # ===========================================================================
        return synthesis_img, self._box_pseudo_gt

    @staticmethod
    def _init_camera(height, K, img_size, device="cpu"):
        """
        Set camera position and parameters.\n
        :param device: device.
        :param height: height of the camera to the ground.
        :param K: 3x3 camera intrinsic matrix.
        :param img_size: tuple(h, w) used for ndc coordinate.
        :return: Instance of PerspectiveCameras.
        """
        # World Coordinate   Camera Coordinate
        #             back to front
        #      z ⊕→ x            ↑ y
        #        ↓ y          x ← ⊕ z
        # camera position in world: (0, h, 0). h is the height of the camera to the ground.
        R, T = look_at_view_transform(eye=((0, -height, 0),), at=((0, -height, 1),), up=((0, -1, 0),), device=device)
        camera = PerspectiveCameras(focal_length=((K[0, 0], K[1, 1]),),
                                    principal_point=((K[0, 2], K[1, 2]),),
                                    in_ndc=False,
                                    image_size=(img_size,),
                                    R=R,
                                    T=T,
                                    device=device)
        return camera

    @staticmethod
    def _init_light(light_type="point", device="cpu", **light_args):
        """light_args = [ambient_color, diffuse_color, specular_color, location]"""
        light = None
        if "ambient" == light_type:
            light = AmbientLights(device=device)
        if "point" == light_type:
            ambient_color = light_args.get("ambient_color") if light_args.get("ambient_color") else (0.5, 0.5, 0.5)
            diffuse_color = light_args.get("diffuse_color") if light_args.get("diffuse_color") else (0.3, 0.3, 0.3)
            specular_color = light_args.get("specular_color") if light_args.get("specular_color") else (0.2, 0.2, 0.2)
            location = light_args.get("location") if light_args.get("location") else (0, 1, 0)
            light = PointLights(ambient_color=(tuple(ambient_color),),
                                diffuse_color=(tuple(diffuse_color),),
                                specular_color=(tuple(specular_color),),
                                location=(tuple(location),),
                                device=device)
        return light

    @staticmethod
    def _init_renderer(quality_rate, img_size, blend_params, camera: PerspectiveCameras, light, device="cpu"):
        """
        Set renderer parameters.\n
        :param quality_rate: quality rate, used for render quality.
        :param img_size: render image size.
        :return: Instance of MeshRenderer.
        """
        # Rasterization Setting
        raster_settings = RasterizationSettings(
            image_size=(img_size[0] * quality_rate, img_size[1] * quality_rate),
            faces_per_pixel=2,
        )
        # Renderer
        renderer = MeshRendererWithMask(
            rasterizer=MeshRasterizer(
                cameras=camera,
                raster_settings=raster_settings
            ),
            shader=Shader(
                device=device,
                cameras=camera,
                lights=light,
                blend_params=blend_params
            )
        )
        return renderer

    @staticmethod
    def _eval_2d_pseudo_gt(mask):
        index = torch.nonzero(mask.squeeze()).T
        y_min = index[0].min().item()
        x_min = index[1].min().item()
        y_max = index[0].max().item()
        x_max = index[1].max().item()
        return [x_min, y_min, x_max, y_max]

    @staticmethod
    def _eval_2d_pseudo_gt_from_bg(render_in_bg, background_color):
        # Calculate Pseudo 2D Box GT
        with torch.no_grad():
            if render_in_bg.requires_grad:
                mesh_in_back_copy = render_in_bg.detach().clone().cpu()
            else:
                mesh_in_back_copy = render_in_bg.clone().cpu()
            # merge rgb value
            mesh_in_back_copy_sum_c = mesh_in_back_copy.sum(dim=2)
            index = torch.nonzero(mesh_in_back_copy_sum_c - np.array(background_color).sum()).T
            y_min = index[0].min()
            x_min = index[1].min()
            y_max = index[0].max()
            x_max = index[1].max()
        return [x_min, y_min, x_max, y_max]

    @staticmethod
    def _get_normalized_img_tensor(image: np.ndarray, device=torch.device("cpu")):
        """
        Get normalized image tensor in RGB type.\n
        :param image: HWC -> RGB -> 0~255 np.ndarray.
        :param device: tensor device.
        :return: Tensor(H, W, 3) of image.
        """
        img = image / 255.0
        return torch.tensor(img, device=device)
