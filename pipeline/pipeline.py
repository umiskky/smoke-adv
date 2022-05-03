from defense.transform import Transform
from pipeline.modules.dataset import Dataset
from pipeline.modules.loss import Loss
from pipeline.modules.sample import Sample
from pipeline.modules.visualization import Visualization
from render.object_loader import ObjectLoader
from render.renderer import Renderer
from render.scenario import Scenario
from render.texture_sticker import TextureSticker
from smoke.smoke import Smoke
from tools.config import Config
from tools.file_utils import update_dic


class Pipeline:

    def __init__(self, args: Config):
        super().__init__()

        self._enable = args.cfg_enable

        # =================== load dataset ====================
        self.dataset = Dataset(args.cfg_dataset)
        # =====================================================

        # =================== load scenario ===================
        if self._enable["scenario"]:
            self.scenario = Scenario(args.cfg_scenario, self.dataset.scenario_indexes)
        else:
            self.scenario = None
        # =====================================================

        # ==================== load object ====================
        if self._enable["object"] and self._enable["renderer"]:
            self.object_loader = ObjectLoader(args.cfg_object)
        else:
            self.object_loader = None
        # =====================================================

        # =================== load stickers ===================
        if self._enable["stickers"] and self._enable["object"] and self._enable["renderer"]:
            self.stickers = TextureSticker(args.cfg_stickers, self.object_loader.textures)
        else:
            self.stickers = None
        # =====================================================

        # =================== load renderer ===================
        if self._enable["renderer"] and self._enable["object"]:
            self.renderer = Renderer(args.cfg_renderer)
        else:
            self.renderer = None
        # =====================================================

        # =================== load defense ====================
        if self._enable["defense"]:
            self.defense = Transform(args.cfg_defense)
        else:
            self.defense = None
        # =====================================================

        # ==================== load smoke =====================
        if self._enable["smoke"]:
            self.smoke = Smoke(args.cfg_smoke)
        else:
            self.smoke = None
        # =====================================================

        # ===================== load loss =====================
        if self._enable["loss"]:
            self.loss = Loss(args.cfg_attack)
        else:
            self.loss = None
        # =====================================================

        # =============== load visualization ==================
        if self._enable["logger"]:
            self.visualization = Visualization(args.cfg_logger)
        else:
            self.visualization = None
        # =====================================================

    def forward(self, sample: Sample):
        # Init
        scenario = scenario_size = mesh = texture = synthesis_img = box3d_branch = loss = None
        box_pseudo_gt = {}

        # Scenario
        if self.scenario is not None:
            scenario, scenario_size = self.scenario.forward(scenario_index=sample.scenario_index)
        # Render Pipeline
        if self.object_loader is not None:
            mesh, box_pseudo_gt_ = self.object_loader.forward(sample)
            box_pseudo_gt = update_dic(box_pseudo_gt_, box_pseudo_gt)
            if self.stickers is not None:
                mesh = self.stickers.forward(mesh, enable_patch_grad=self._enable["attack"])
            if self.renderer is not None:
                # synthesis_img [0.0, 255.0]
                synthesis_img, box_pseudo_gt_ = self.renderer.forward(mesh, scenario, sample)
                box_pseudo_gt = update_dic(box_pseudo_gt_, box_pseudo_gt)
        # Smoke Pipeline
        if self.smoke is not None:
            purifier_img = None
            if self.renderer is not None:
                if self.defense is not None:
                    purifier_img = self.defense.forward(synthesis_img)
                else:
                    purifier_img = synthesis_img
            elif scenario is not None:
                if self.defense is not None:
                    purifier_img = self.defense.forward(scenario)
                else:
                    purifier_img = scenario
            box3d_branch, _ = self.smoke.forward(purifier_img, sample)
            if self.loss is not None:
                loss = self.loss.forward(box_pseudo_gt=box_pseudo_gt,
                                         box3d_branch=box3d_branch,
                                         smoke=self.smoke)
                return loss

        # Result Setting
        result_list = [box3d_branch, synthesis_img, scenario]
        for result in result_list:
            if result is not None:
                return result, box_pseudo_gt
        return None
