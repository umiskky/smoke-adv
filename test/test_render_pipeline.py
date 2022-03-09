import os

import cv2
import torch
from matplotlib import pyplot as plt
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import look_at_view_transform, PerspectiveCameras, \
    RasterizationSettings, PointLights, MeshRenderer, MeshRasterizer, SoftPhongShader, AmbientLights, \
    HardPhongShader, BlendParams
from pytorch3d.transforms import Transform3d, Scale, Translate, RotateAxisAngle, se3_exp_map
import torch.nn.functional as F

if __name__ == '__main__':
    # Setup
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    # Set paths
    DATA_DIR = "../data/objects/"
    obj_filename = os.path.join(DATA_DIR, "man/man.obj")
    # Load obj file
    mesh = load_objs_as_meshes([obj_filename], device=device)

    # World Coordinate   Camera Coordinate
    #             back to front
    #        ^ y                ^ y
    #        |                  |
    #      z ⊙--> x      x <-- ⊕ z
    # <=================== model matrix ===================>
    scale_rate = 0.01
    model_matrix = Transform3d(device=device)
    model_matrix = model_matrix.compose(Scale(scale_rate, device=device))

    # <====================================================>

    # <=================== model2 matrix ===================>
    model_matrix = model_matrix.compose(RotateAxisAngle(0, "X", device=device))
    model_matrix = model_matrix.compose(RotateAxisAngle(20, "Y", device=device))
    model_matrix = model_matrix.compose(RotateAxisAngle(0, "Z", device=device))
    # model_matrix = model_matrix.compose(Translate(4, 0, -6, device=device))
    model_matrix = model_matrix.compose(Translate(2, 0.1, -4, device=device))

    print(model_matrix.get_matrix())
    print(RotateAxisAngle(90, "Y", device=device).get_matrix())
    verts_list = getattr(mesh, "_verts_list")
    verts_list[0][:] = model_matrix.transform_points(verts_list[0])
    print(id(verts_list) == id(getattr(mesh, "_verts_list")))
    # <=====================================================>

    # <==================== view matrix ====================>
    #      ___________________________                            ^ y_w
    #   _ /   ___/~~~    /------------\                           |
    # _|_/_______________|______|____|__\_____________            |
    # \ _________________|____-_|-______|_____________)           |
    #  <____//   \|______|______|_______|_//   \)_____>  z_w <--- ⊙ x_w
    #        \___/                         \___/
    # camera position(0.48, 1.65, 1.68)
    R, T = look_at_view_transform(eye=((0.48, 1.65, 1.68),), at=((0.48, 1.65, 0),), device=device)
    print(R)
    print(T)
    # <=====================================================>

    # CV Camera
    camera_sfm = PerspectiveCameras(focal_length=((7.070493000000e+02, 7.070493000000e+02),),
                                    principal_point=((6.040814000000e+02, 1.805066000000e+02),),
                                    in_ndc=False,
                                    image_size=((375, 1242),),
                                    R=R,
                                    T=T,
                                    device=device)
    # Light Setting
    # lights = PointLights(device=device, location=[[5.0, 10.0, -10.0]])
    lights = AmbientLights(device=device)

    # Rasterization Setting
    raster_settings = RasterizationSettings(
        image_size=(375*1*6, 1242*1*6),
    )

    blendParams = BlendParams(
        background_color=(0.0, 0.0, 0.0)
    )
    # Renderer
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera_sfm,
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(
            device=device,
            cameras=camera_sfm,
            lights=lights,
            blend_params=blendParams
        )
    )

    # Plot Result
    images = renderer(mesh)
    images = F.adaptive_avg_pool2d(images[0, ..., :3].cpu().permute(2, 0, 1).unsqueeze(0), (375, 1242))
    images = images[0, :].permute(1, 2, 0)
    plt.figure(figsize=(12.42, 3.75))
    # plt.imshow(images[0, ..., :3].cpu().numpy())
    plt.imshow(images.numpy())
    plt.axis("off")
    plt.show()
    # plt.imsave("../data/objects/man/test_render_pipeline.jpg", images[0, ..., :3].cpu().numpy())
    plt.imsave("../data/objects/man/test_render_pipeline.jpg", images.numpy())

    # Synthesis Result
    ori_img = cv2.imread("../data/examples/000448.png")
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB) / 255.0
    ori_img_tensor = torch.tensor(ori_img)

    mask_indexes = torch.nonzero(images.sum(2))
    mask_indexes = mask_indexes[:, 0], mask_indexes[:, 1]
    mask = torch.ones(images.shape)
    mask = mask.index_put(mask_indexes, torch.tensor([0.0, 0.0, 0.0]))
    mask_img = ori_img_tensor * mask
    synthesis_img = mask_img + images

    plt.figure(figsize=(12.42, 3.75))
    # plt.imshow(images[0, ..., :3].cpu().numpy())
    plt.imshow(synthesis_img.numpy())
    plt.axis("off")
    plt.show()
    plt.imsave("../data/objects/man/test_render_synthesis.jpg", synthesis_img.numpy())
