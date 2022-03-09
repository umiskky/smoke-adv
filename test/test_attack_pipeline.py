import torch
from pytorch3d.renderer import PointLights

from render.object_loader import ObjectLoader
from render.renderer import Renderer
from smoke.dataset.waymo import getKMatrix
from tools.visualization.vis_render import vis_img


class AttackPipeline:
    def __init__(self, device=torch.device(torch.device("cpu"))) -> None:
        self.device = device
        self.object_loader = None
        self.renderer = None
        self.texture_sticker = None
        self.smoke = None

    def initial(self):
        pass


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    calib_idx = "0002045"
    scenario = "../data/datasets/waymo/kitti_format/training/image_0/" + calib_idx + ".png"
    scenario_tensor = Renderer.get_rgb_normalized_img_tensor(scenario, device=device)

    # scenario_tensor.requires_grad_(True)

    img_size = scenario_tensor.shape

    ap = AttackPipeline(device)

    # Init Object
    # =========================================================================================
    ol = ObjectLoader(device, obj_path="../data/objects/man/man.obj")
    ol.set_model_matrix(rotation=(0, 185, 0), translate=(4, 0, -10), scale_rate=0.0085)
    ol.load()

    # texture = ol.mesh.textures
    # texture_tensor = getattr(texture, "_maps_padded")
    # texture_tensor.requires_grad_(True)

    ap.object_loader = ol
    # =========================================================================================

    # Init Render
    # =========================================================================================
    rd = Renderer(device)
    rd.set_camera(height=1.85, K=getKMatrix(calib_idx), img_size=(img_size[0], img_size[1]))
    # rd.set_light()
    rd.light = PointLights(ambient_color=((0.28, 0.28, 0.28),),
                           diffuse_color=((0.25, 0.25, 0.25),),
                           specular_color=((0, 0, 0),),
                           location=((10, 8, -6),),
                           device=device)
    rd.set_render(quality_rate=1, img_size=(img_size[0], img_size[1]))
    ap.renderer = rd
    # =========================================================================================

    # Init Texture Sticker
    # =========================================================================================
    # ts = TextureSticker(device)
    # ts.sticker = ts.generate_uniform_sticker(size=1200)
    # ap.texture_sticker = ts
    # =========================================================================================

    # Init Smoke Model
    # =========================================================================================
    # smoke = Smoke(device)
    # smoke.k = getKMatrix(calib_idx)
    # ap.smoke = smoke
    # =========================================================================================

    # Construction of pipeline
    # =========================================================================================
    print("Synthesis Image ...")
    synthesis_img = ap.renderer.render(mesh=ap.object_loader.mesh, target=scenario_tensor, vis=True, save=False,
                                       save_path="../data/results/attack/" + calib_idx + "_mesh_in_back" + ".png")

    # test = synthesis_img.sum()
    # test.backward()
    # print(getattr(ap.object_loader.mesh.textures, "_maps_padded").grad)

    vis_img(synthesis_img, (synthesis_img.shape[1] / 100, synthesis_img.shape[0] / 100))

    # print("Smoke Inference ...")
    # synthesis_img_t, ori_img_size, _ = transform_input_tensor(synthesis_img, device)
    # smoke.forward(synthesis_img_t, ori_img_size)
    # obstacle_list = Obstacle.decode(smoke.box3d_branch, input_utils.getIntrinsicMatrix(False, smoke.k),
    #                                 ori_img_size)
    # =========================================================================================

    # Visualize Detection Result
    # =========================================================================================
    # synthesis_img_vis = synthesis_img.clone().int().cpu().numpy().astype(np.uint8)
    # # save_img(synthesis_img_vis, "../data/results/attack/" + calib_idx + "_synthesis" + ".png")
    #
    # img_draw_3d = draw_3d_boxes(synthesis_img_vis, obstacle_list)
    # vis_img(img_draw_3d, (img_draw_3d.shape[1] / 100, img_draw_3d.shape[0] / 100))
    # # save_img(img_draw_3d, "../data/results/attack/" + calib_idx + "_3d_synthesis" + ".png")
    #
    # img_draw_2d = draw_2d_boxes(synthesis_img_vis, obstacle_list)
    # vis_img(img_draw_2d, (img_draw_2d.shape[1] / 100, img_draw_2d.shape[0] / 100))
    # # save_img(img_draw_2d, "../data/results/attack/" + calib_idx + "_2d_synthesis" + ".png")
    # =========================================================================================

    pass
