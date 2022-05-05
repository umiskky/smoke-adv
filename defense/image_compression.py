from albumentations import ImageOnlyTransform
from albumentations import functional as F


class JpegCompression(ImageOnlyTransform):

    def __init__(
            self,
            quality=100,
            always_apply=False,
            p=0.5,
    ):
        super(JpegCompression, self).__init__(always_apply, p)
        self.quality = max(min(int(quality), 100), 0)

    def apply(self, image, quality=100, image_type=".jpg", **params):
        if not image.ndim == 2 and image.shape[-1] not in (1, 3, 4):
            raise TypeError("ImageCompression transformation expects 1, 3 or 4 channel images.")
        return F.image_compression(image, quality, image_type)

    def get_params(self):
        image_type = ".jpg"
        return {
            "quality": self.quality,
            "image_type": image_type,
        }

    def get_transform_init_args(self):
        return {
            "quality": self.quality
        }
