import os
import os.path as osp

import numpy as np
import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.transforms import Transform3d, Scale, RotateAxisAngle, Translate

from pipeline.modules.sample import Sample


class ObjectLoader:
    def __init__(self, args: dict) -> None:
        self._device = args["device"]
        self._mesh = self.load(obj_path=osp.join(os.getenv("project_path"),
                                                 args["obj_path"]),
                               device=self._device)
        self._verts: torch.Tensor = getattr(self._mesh, "_verts_list")[0].clone()
        self._textures: torch.Tensor = getattr(self._mesh.textures, "_maps_padded").clone()
        self._box_pseudo_gt = {"3d": {"dimensions": args["size"]}}
        self._rotation_relative = args["rotation_relative"]

    # TODO relative rotation
    def forward(self, sample: Sample):
        """
        Generate new mesh instance with transformed in data.\n
        :param sample: instance of Class Sample.
        :return: mesh with transformed.
        """
        model_transform = self.get_model_matrix(rotation=tuple(sample.rotation),
                                                translate=tuple(sample.location),
                                                scale_rate=sample.scale,
                                                device=self._device,
                                                relative=self._rotation_relative)
        self.apply_model_matrix(self._mesh, model_transform, self._verts.clone())
        return self._mesh, self._box_pseudo_gt

    @property
    def textures(self):
        return self._textures.clone()

    @staticmethod
    def load(obj_path: str, device="cpu"):
        mesh = load_objs_as_meshes([obj_path], device=device)
        return mesh

    @staticmethod
    def apply_model_matrix(mesh, model_matrix, verts):
        # Apply transform to mesh
        setattr(mesh, "_verts_list", [model_matrix.transform_points(verts)])
        # Reset All Caches
        setattr(mesh, "_verts_normals_packed", None)
        setattr(mesh, "_verts_packed", None)
        setattr(mesh, "_verts_packed_to_mesh_idx", None)
        setattr(mesh, "_verts_padded", None)
        setattr(mesh, "_verts_padded_to_packed_idx", None)

    @staticmethod
    def get_model_matrix(rotation: tuple = None, translate: tuple = None, scale_rate=0.01, relative=False,
                         device="cpu"):
        """
        Set transform to mesh. In order of scale, rotation, translate.\n
        :param relative: rotation relative or not.
        :param device: device.
        :param rotation: tuple of rotation. (X, Y, Z).
        :param translate: tuple of translate. (X, Y, Z).
        :param scale_rate: scale rate. scalar.
        :return: None
        """
        # World Coordinate    Model Coordinate
        #             back to front
        #      z ⊕→ x            ↑ y
        #        ↓ y             z ⊙→ x
        transform = Transform3d(device=device)

        # Model Coordinate To World Coordinate
        transform = transform.compose(RotateAxisAngle(angle=180, axis="X", device=device))
        # Model Transform
        transform = transform.compose(Scale(scale_rate, device=device))
        # relative rotation
        # camera is locate on y-negative-axis, looking at z-axis
        # 3d object stand at x-o-z plane, looking at z-negative-axis
        # assume that the angle is zero when 3d object faces the camera
        if relative:
            transform = transform.compose(RotateAxisAngle(
                angle=(np.arctan2(translate[0], translate[2]) * 180.0 / np.pi).item(),
                axis="Y", device=device))
        # rotation
        transform = transform.compose(RotateAxisAngle(angle=rotation[0], axis="X", device=device))
        transform = transform.compose(RotateAxisAngle(angle=rotation[1], axis="Y", device=device))
        transform = transform.compose(RotateAxisAngle(angle=rotation[2], axis="Z", device=device))
        transform = transform.compose(Translate(x=translate[0], y=translate[1], z=translate[2], device=device))
        model_matrix = transform
        return model_matrix
