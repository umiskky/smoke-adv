from albumentations import ImageOnlyTransform
from albumentations import functional as F


class MedianBlur(ImageOnlyTransform):
    def __init__(self, kernel_size=3, always_apply=False, p=0.5):
        super(MedianBlur, self).__init__(always_apply, p)
        self.kernel_size = kernel_size

    def apply(self, image, kernel_size=3, **params):
        return F.median_blur(image, kernel_size)

    def get_params(self):
        return {"kernel_size": self.kernel_size}

    def get_transform_init_args_names(self):
        return "kernel_size",
