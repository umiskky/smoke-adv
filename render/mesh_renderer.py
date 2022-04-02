from pytorch3d.renderer import MeshRenderer


class MeshRendererWithMask(MeshRenderer):

    def __init__(self, rasterizer, shader) -> None:
        super().__init__(rasterizer, shader)

    def forward(self, meshes_world, **kwargs):
        fragments = self.rasterizer(meshes_world, **kwargs)
        images, mask = self.shader(fragments, meshes_world, **kwargs)
        return images, mask
