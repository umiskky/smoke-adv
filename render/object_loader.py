import os.path as osp

import torch
import torch.nn as nn
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.transforms import Transform3d, Scale, RotateAxisAngle, Translate
# import pytorch3d.renderer.mesh.textures

class ObjectLoader(nn.Module):
    def __init__(self, args: dict, global_args: dict) -> None:
        super().__init__()
        self.device = torch.device(args["device"])
        self.obj_path = osp.join(global_args["project_path"], args["obj_path"])
        self.mesh = None
        self.model_matrix = None
        model_matrix: dict = args["model_matrix"]
        self.set_model_matrix(rotation=tuple(model_matrix["rotation"]),
                              translate=tuple(model_matrix["translate"]),
                              scale_rate=model_matrix["scale"])
        self.load()

    def forward(self):
        return self.mesh

    def load(self):
        assert self.model_matrix is not None
        self.mesh = load_objs_as_meshes([self.obj_path], device=self.device)
        # Apply transform to mesh
        verts_list = getattr(self.mesh, "_verts_list")
        verts_list[0][:] = self.model_matrix.transform_points(verts_list[0])

    def set_model_matrix(self, rotation=None, translate=None, scale_rate=0.01):
        """
        Set transform to mesh. In order of scale, rotation, translate.\n
        :param rotation: tuple of rotation. (X, Y, Z).
        :param translate: tuple of translate. (X, Y, Z).
        :param scale_rate: scale rate. scalar.
        :return: None
        """
        # World Coordinate
        #  back to front
        #        ^ y
        #        |
        #      z ⊙--> x
        transform = Transform3d(device=self.device)
        transform = transform.compose(Scale(scale_rate, device=self.device))
        transform = transform.compose(RotateAxisAngle(rotation[0], "X", device=self.device))
        transform = transform.compose(RotateAxisAngle(rotation[1], "Y", device=self.device))
        transform = transform.compose(RotateAxisAngle(rotation[2], "Z", device=self.device))
        transform = transform.compose(Translate(translate[0], translate[1], translate[2], device=self.device))
        self.model_matrix = transform