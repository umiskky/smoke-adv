import os.path as osp

from comet_ml import Experiment

from render.renderer import Renderer
from render.scenario import Scenario
from render.texture_sticker import TextureSticker
from smoke.smoke import Smoke
from tools.utils import makedirs
from tools.visualization.vis_detection import draw_3d_boxes, draw_2d_boxes
from tools.visualization.vis_render import plot_img, save_img, log_img


class Visualization:
    def __init__(self, args: dict, global_args: dict):
        self.args = args
        self.global_args = global_args
        self.timestamp = args["timestamp"]
        self.experiment_path = osp.join(global_args["project_path"], "data/results", self.timestamp)
        self.save_dir = osp.join(global_args["project_path"], "data/results", self.timestamp, "visualization")
        self.save = args["save"]
        # Init Logger Name
        self.logger: Experiment = args["logger"]

        if self.save:
            makedirs(self.save_dir)
            self.init_dir()
        # counter
        self.counter = 0
        # enable for figure plot only once
        self.once = True

    def vis(self, scenario: Scenario, renderer: Renderer, stickers: TextureSticker, smoke: Smoke):
        if self.once:
            if scenario is not None and self.args["scenario"] and not len(scenario.visualization) == 0:
                # 0-255 RGB HWC uint8
                scenario_plot = scenario.visualization["scenario"]
                if self.args["plot"]:
                    plot_img(scenario_plot, "scenario")
                if self.args["save"]:
                    saving_path = osp.join(self.save_dir, "scenario.png")
                    save_img(scenario_plot, saving_path)
                if self.logger is not None:
                    log_img(logger=self.logger, image=scenario_plot, img_name="scenario", step=self.counter)
            self.once = False

        if self.args["sticker"]:
            # TODO
            if self.args["plot"]:
                pass
            if self.args["save"]:
                pass

        if stickers is not None and self.args["texture"] and not len(stickers.visualization) == 0:
            # 0~1.0 RGB HWC float32
            texture_plot = stickers.visualization["texture"]
            if self.args["plot"]:
                plot_img(texture_plot, "texture")
            if self.args["save"]:
                saving_path = osp.join(self.save_dir, "sticker", "%05d" % self.counter + "_step_texture.png")
                save_img(texture_plot, saving_path)
            if self.logger is not None:
                log_img(logger=self.logger, image=texture_plot,
                        img_name="sticker_" + "%05d" % self.counter + "_step_texture",
                        step=self.counter)

        if renderer is not None and self.args["render_bg"] and not len(renderer.visualization) == 0:
            # 0~1.0 RGB HWC float32
            render_bg_plot = renderer.visualization["render_bg"]
            if self.args["plot"]:
                plot_img(render_bg_plot, "render in background")
            if self.args["save"]:
                saving_path = osp.join(self.save_dir, "render_bg", "%05d" % self.counter + "_step_render_bg.png")
                save_img(render_bg_plot, saving_path)
            if self.logger is not None:
                log_img(logger=self.logger,
                        image=render_bg_plot,
                        img_name="render_bg_" + "%05d" % self.counter + "_step_render_bg",
                        step=self.counter)

        if renderer is not None and self.args["render_scenario"] and not len(renderer.visualization) == 0:
            # 0~1.0 RGB HWC float64
            render_scenario_plot = renderer.visualization["render_scenario"]
            if self.args["plot"]:
                plot_img(render_scenario_plot, "render in scenario")
            if self.args["save"]:
                saving_path = osp.join(self.save_dir, "render_scenario",
                                       "%05d" % self.counter + "_step_render_scenario.png")
                save_img(render_scenario_plot, saving_path)
            if self.logger is not None:
                log_img(logger=self.logger,
                        image=render_scenario_plot,
                        img_name="render_scenario_" + "%05d" % self.counter + "_step_render_scenario",
                        step=self.counter)

        obstacle_list = smoke.visualization["detection"] if smoke is not None else None
        # 0~255 RGB HWC uint8
        scenario_image = scenario.visualization["scenario"] if scenario is not None else None
        # 0~1.0 RGB HWC float64
        render_scenario_image = renderer.visualization["render_scenario"] if renderer is not None else None
        if smoke is not None and self.args["detection_3d"] and not len(smoke.visualization) == 0:
            # 0~255 RGB HWC uint8
            detection_3d_plot = draw_3d_boxes(render_scenario_image if render_scenario_image is not None
                                              else scenario_image, obstacle_list)
            if self.args["plot"]:
                plot_img(detection_3d_plot, "detection 3d")
            if self.args["save"]:
                saving_path = osp.join(self.save_dir, "detection_3d", "%05d" % self.counter + "_step_detection_3d.png")
                save_img(detection_3d_plot, saving_path)
            if self.logger is not None:
                log_img(logger=self.logger,
                        image=detection_3d_plot,
                        img_name="detection_3d_" + "%05d" % self.counter + "_step_detection_3d",
                        step=self.counter)

        if smoke is not None and self.args["detection_2d"] and not len(smoke.visualization) == 0:
            # 0~255 RGB HWC uint8
            detection_2d_plot = draw_2d_boxes(render_scenario_image if render_scenario_image is not None
                                              else scenario_image, obstacle_list)
            if self.args["plot"]:
                plot_img(detection_2d_plot, "detection 2d")
            if self.args["save"]:
                saving_path = osp.join(self.save_dir, "detection_2d", "%05d" % self.counter + "_step_detection_2d.png")
                save_img(detection_2d_plot, saving_path)
            if self.logger is not None:
                log_img(logger=self.logger,
                        image=detection_2d_plot,
                        img_name="detection_2d_" + "%05d" % self.counter + "_step_detection_2d",
                        step=self.counter)

        # if self.args["save"]:
        #     self.counter += 1

    def init_dir(self):
        if self.args["sticker"]:
            makedirs(self.save_dir, "sticker")
        if self.args["texture"]:
            makedirs(self.save_dir, "texture")
        if self.args["render_bg"]:
            makedirs(self.save_dir, "render_bg")
        if self.args["render_scenario"]:
            makedirs(self.save_dir, "render_scenario")
        if self.args["detection_3d"]:
            makedirs(self.save_dir, "detection_3d")
        if self.args["detection_2d"]:
            makedirs(self.save_dir, "detection_2d")
