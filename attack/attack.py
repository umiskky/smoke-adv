import torch.nn as nn

from render.object_loader import ObjectLoader
from render.renderer import Renderer
from render.scenario import Scenario
from render.texture_sticker import TextureSticker
from smoke.smoke import Smoke
from tools.config import Config
from tools.visualization.visualization import Visualization


class Attack(nn.Module):
    def __init__(self, args: Config):
        super().__init__()

        # =================== load scenario ===================
        self.scenario = Scenario(args.cfg_scenario, args.cfg_global)
        # write to other args
        args.cfg_renderer["camera"]["K"] = self.scenario.K
        args.cfg_renderer["scenario_size"] = self.scenario.scenario_size
        args.cfg_smoke["K"] = self.scenario.K
        args.cfg_smoke["scenario_size"] = self.scenario.scenario_size
        # =====================================================

        # ==================== load object ====================
        if args.cfg_object["switch_on"] and args.cfg_renderer["switch_on"]:
            self.object_loader = ObjectLoader(args.cfg_object, args.cfg_global)
        else:
            self.object_loader = None
        # =====================================================

        # =================== load renderer ===================
        if args.cfg_renderer["switch_on"]:
            self.renderer = Renderer(args.cfg_renderer)
        else:
            self.renderer = None
        # =====================================================

        # =================== load stickers ===================
        if args.cfg_stickers["switch_on"]:
            self.stickers = TextureSticker(args.cfg_stickers, args.cfg_global)
        else:
            self.stickers = None
        # =====================================================

        # =================== load defense ====================
        if args.cfg_defense["switch_on"]:
            self.defense = None
        else:
            self.defense = None
        # =====================================================

        # ==================== load smoke =====================
        if args.cfg_smoke["switch_on"]:
            self.smoke = Smoke(args.cfg_smoke, args.cfg_global)
        else:
            self.smoke = None
        # =====================================================

        # ==================== load attack ====================
        # TODO
        if args.cfg_attack["switch_on"]:
            self.attack = None
        else:
            self.attack = None
        # =====================================================

        # =============== load visualization ==================
        self.visualization = Visualization(args.cfg_visualization, args.cfg_global)
        # =====================================================

    def forward(self):
        # Init
        mesh = texture = synthesis_img = box3d_branch = feat_branch = None
        K, scenario, scenario_size = self.scenario.forward()
        # Render Pipeline
        if self.object_loader is not None:
            mesh, texture = self.object_loader.forward()
            if self.stickers is not None:
                mesh = self.stickers.forward(mesh)
            if self.renderer is not None:
                synthesis_img = self.renderer.forward(mesh, scenario)
        # Smoke Pipeline
        if self.smoke is not None:
            if self.renderer is not None:
                box3d_branch, feat_branch = self.smoke.forward(synthesis_img)
            else:
                box3d_branch, feat_branch = self.smoke.forward(scenario)

        # Visualization Pipeline
        self.visualization.vis(scenario=self.scenario,
                               renderer=self.renderer,
                               stickers=self.stickers,
                               smoke=self.smoke)

        # Result Setting
        result_list = [box3d_branch, synthesis_img, scenario]
        for result in result_list:
            if result is not None:
                return result
        return None
