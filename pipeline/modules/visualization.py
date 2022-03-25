import os
import os.path as osp

from render.renderer import Renderer
from render.scenario import Scenario
from render.texture_sticker import TextureSticker
from smoke.obstacle import Obstacle
from smoke.smoke import Smoke
from tools.file_utils import makedirs, state_saving
from tools.vis_utils import draw_3d_boxes, draw_2d_boxes, plot_img, save_img, log_img


class Visualization:
    def __init__(self, args: dict):
        self._timestamp = os.getenv("timestamp")
        self._project_path = os.getenv("project_path")

        self._confidence_threshold = args["common"]["confidence_threshold"]

        # comet logger and python logger
        self._enable_comet = args["comet"]["enable"]
        self._comet_content = args["comet"]["vis_content"]
        self._logger_comet, self._logger_console = args["logger"]
        # vis in sci view
        self._enable_vis_plt = args["local"]["vis_plt"]
        self._plt_content = args["local"]["plt_content"]
        # vis offline in local computer
        self._enable_vis_offline = args["local"]["vis_offline"]
        self._offline_dir = osp.join(self._project_path, "data/results", self._timestamp, "visualization")
        self._off_content = args["local"]["off_content"]

        if self._enable_vis_offline:
            makedirs(self._offline_dir)
            self._init_dir(args["local"]["off_content"])
        # step & epoch
        self._step = -1
        self._epoch = -1
        # enable for figure plot only once
        self._once = True

    def vis(self, scenario_index, epoch, step, scenario: Scenario, renderer: Renderer, stickers: TextureSticker, smoke: Smoke):
        """
        Vis For Pipeline.\n
        :param scenario_index: scenario index.
        :param epoch: epoch.
        :param step: step.
        :param scenario: instance of Scenario.
        :param renderer: instance of Renderer.
        :param stickers: instance of TextureSticker.
        :param smoke: instance of Smoke.
        :return: None
        """
        self._step = step
        # Vis only once
        if self._once:
            self._vis_scenario(scenario)
            self._once = False
        # Vis every epoch
        if self._epoch < epoch:
            self._vis_texture(stickers)
            self._epoch = epoch
        # Vis every step
        self._vis_render_bg(renderer, scenario_index)
        self._vis_render_scenario(renderer, scenario_index)
        self._vis_detection(smoke, scenario_index)

    def _vis_scenario(self, scenario: Scenario):
        if scenario is not None and len(scenario.visualization) > 0:
            # 0-255 RGB HWC uint8
            scenario_plot_dict = scenario.visualization.get("scenarios")
            if scenario_plot_dict is None:
                return None
            for index in scenario_plot_dict.keys():
                if self._enable_vis_plt and "scenario" in self._plt_content:
                    plot_img(scenario_plot_dict.get(index), "scenario_%s" % index)
                if self._enable_vis_offline and "scenario" in self._off_content:
                    saving_path = osp.join(self._offline_dir, "scenario/", "scenario_%s.png" % index)
                    save_img(scenario_plot_dict.get(index), saving_path)
                if self._enable_comet and "scenario" in self._comet_content:
                    log_img(logger=self._logger_comet, image=scenario_plot_dict.get(index),
                            img_name="scenario_%s" % index, step=-1)

    def _vis_texture(self, stickers: TextureSticker):
        if stickers is not None and len(stickers.visualization) > 0:
            # 0~1.0 RGB HWC float32
            texture_plot = stickers.visualization.get("texture")
            if texture_plot is None:
                return None
            if self._enable_vis_plt and "texture" in self._plt_content:
                plot_img(texture_plot, "texture_%04d-epoch" % self._epoch)
            if self._enable_vis_offline and "texture" in self._off_content:
                saving_path = osp.join(self._offline_dir, "sticker", "%04d" % self._epoch + "-epoch_texture.png")
                save_img(texture_plot, saving_path)
            if self._enable_comet and "texture" in self._comet_content:
                log_img(logger=self._logger_comet, image=texture_plot,
                        img_name="%04d" % self._epoch + "-epoch_texture",
                        step=self._epoch)

    def _vis_render_bg(self, renderer: Renderer, scenario_index):
        if renderer is not None and len(renderer.visualization) > 0:
            # 0~1.0 RGB HWC float32
            render_bg_plot = renderer.visualization["render_bg"]
            if render_bg_plot is None:
                return None
            epoch_step = "%04d-epoch_%04d-step" % (self._epoch, self._step)
            if self._enable_vis_plt and "render_bg" in self._plt_content:
                plot_img(render_bg_plot, "render_bg_" + epoch_step)
            if self._enable_vis_offline and "render_bg" in self._off_content:
                saving_path = osp.join(self._offline_dir, scenario_index + "_" + epoch_step + "_render_bg.png")
                save_img(render_bg_plot, saving_path)
            if self._enable_comet and "render_bg" in self._comet_content:
                log_img(logger=self._logger_comet,
                        image=render_bg_plot,
                        img_name=scenario_index + "_" + epoch_step + "_render_bg",
                        step=self._epoch)

    def _vis_render_scenario(self, renderer: Renderer, scenario_index):
        if renderer is not None and len(renderer.visualization) > 0:
            # 0~1.0 RGB HWC float64
            render_scenario_plot = renderer.visualization.get("render_scenario")
            if render_scenario_plot is None:
                return None
            epoch_step = "%04d-epoch_%04d-step" % (self._epoch, self._step)
            if self._enable_vis_plt and "render_scenario" in self._plt_content:
                plot_img(render_scenario_plot, "render_scenario_" + epoch_step)
            if self._enable_vis_offline and "render_scenario" in self._off_content:
                saving_path = osp.join(self._offline_dir, "render_scenario",
                                       scenario_index + "_" + epoch_step + "_render_scenario.png")
                save_img(render_scenario_plot, saving_path)
            if self._enable_comet and "render_scenario" in self._comet_content:
                log_img(logger=self._logger_comet,
                        image=render_scenario_plot,
                        img_name=scenario_index + "_" + epoch_step + "_render_scenario",
                        step=self._epoch)

    def _vis_detection(self, smoke: Smoke, scenario_index):
        obstacle_list = Obstacle.decode(box3d_branch_data=smoke.visualization.get("detection"),
                                        k=smoke.visualization.get("K"),
                                        confidence_score=self._confidence_threshold,
                                        ori_img_size=smoke.visualization.get("scenario_size"))

        if smoke is not None and len(smoke.visualization) > 0:
            # 0~255 RGB HWC uint8
            detection_3d_plot = draw_3d_boxes(smoke.visualization.get("scenario"), obstacle_list)
            # 0~255 RGB HWC uint8
            detection_2d_plot = draw_2d_boxes(smoke.visualization.get("scenario"), obstacle_list)

            epoch_step = "%04d-epoch_%04d-step" % (self._epoch, self._step)

            if self._enable_vis_plt:
                if "detection_3d" in self._plt_content:
                    plot_img(detection_3d_plot, "detection_3d_" + epoch_step)
                if "detection_2d" in self._plt_content:
                    plot_img(detection_2d_plot, "detection_2d_" + epoch_step)
            if self._enable_vis_offline:
                if "detection_3d" in self._off_content:
                    saving_path = osp.join(self._offline_dir, "detection_3d",
                                           scenario_index + "_" + epoch_step + "_detection_3d.png")
                    save_img(detection_3d_plot, saving_path)
                if "detection_2d" in self._off_content:
                    saving_path = osp.join(self._offline_dir, "detection_2d",
                                           scenario_index + "_" + epoch_step + "_detection_2d.png")
                    save_img(detection_2d_plot, saving_path)
            if self._enable_comet:
                if "detection_3d" in self._comet_content:
                    log_img(logger=self._logger_comet,
                            image=detection_3d_plot,
                            img_name=scenario_index + "_" + epoch_step + "_detection_3d",
                            step=self._epoch)
                if "detection_2d" in self._comet_content:
                    log_img(logger=self._logger_comet,
                            image=detection_2d_plot,
                            img_name=scenario_index + "_" + epoch_step + "_detection_2d",
                            step=self._epoch)

    def save_patch(self, epoch, loss_epoch, stickers: TextureSticker):
        if stickers.patch.requires_grad:
            save_patch = stickers.patch.detach().clone().cpu()
        else:
            save_patch = stickers.patch.clone().cpu()
        # save texture patch
        if self._enable_vis_offline and "patch" in self._off_content:
            state_dict = {"patch": save_patch, "epoch": epoch, "loss": loss_epoch}
            state_saving(state=state_dict, epoch=epoch, loss=loss_epoch, path=self._offline_dir)

    def _init_dir(self, contents: list):
        for content in contents:
            makedirs(self._offline_dir, content)
