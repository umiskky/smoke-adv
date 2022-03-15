import os

import torch
from matplotlib import pyplot as plt
from pytorch3d.io import load_objs_as_meshes

from render import TextureSticker


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
        device = torch.device("cuda:1")
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

    ts = TextureSticker(device)
    ts.sticker = TextureSticker.generate_uniform_tensor(size=1200, require_grad=True, device=device)
    print(ts.sticker)

    ts.apply_hls_sticker(mesh, (x_l, y_l), intensity=0.8)
    display_texture(mesh.textures.maps_padded().detach())
    pass


