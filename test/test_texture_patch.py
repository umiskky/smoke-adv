import os

import torch
from matplotlib import pyplot as plt
from pytorch3d.io import load_objs_as_meshes


def display_texture(texture, fig_size=5):
    plt.figure(figsize=(fig_size, fig_size))
    plt.imshow(texture.squeeze().cpu().numpy())
    plt.grid("off")
    plt.axis('off')
    plt.show()


def add_patch_texture(texture, p, x_l, y_l):
    patch = torch.zeros_like(texture)
    mask = torch.ones_like(texture)
    size_patch = p.shape[1]
    patch[:, y_l:y_l + size_patch, x_l:x_l + size_patch, :] = p
    mask[:, y_l:y_l + size_patch, x_l:x_l + size_patch, :] = 0
    return (mask * texture) + patch


if __name__ == "__main__":
    # Setup
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    obj_filename = os.path.join("../data/objects", "man/man.obj")
    # Load obj file
    mesh = load_objs_as_meshes([obj_filename], device=device)
    texture_image = mesh.textures.maps_padded()

    # 1. Get Original Texture
    display_texture(texture_image)
    # New Texture
    size_patch = 1200
    x_l = 4600
    y_l = 5600
    print('Patch')
    p = torch.rand((size_patch, size_patch, 3))
    plt.imshow(p.numpy())
    plt.show()
    # 2. Create New Texture
    applied_patch = add_patch_texture(texture_image, p, x_l, y_l)[0]
    display_texture(applied_patch)

    # 3. Replace old texture in original mesh??
    ################################# 方法一 #################################
    # texture = mesh.textures
    # if hasattr(texture, "_faces_uvs_list") and hasattr(texture, "_verts_uvs_list"):
    #     texture_new = TexturesUV(applied_patch[None], getattr(texture, "_faces_uvs_list"),
    #                              getattr(texture, "_verts_uvs_list"))
    #     mesh.textures = texture_new
    # display_texture(mesh.textures.maps_padded())

    ################################# 方法二 #################################
    # texture = mesh.textures
    # setattr(texture, "_maps_padded", applied_patch[None])
    # display_texture(mesh.textures.maps_padded())

    ################################# 方法三 #################################
    # texture = mesh.textures
    # texture_tensor = getattr(texture, "_maps_padded")
    # print(texture_tensor.shape)
    # texture_tensor[:] = add_patch_texture(texture_tensor, p, x_l, y_l)
    # display_texture(mesh.textures.maps_padded())
