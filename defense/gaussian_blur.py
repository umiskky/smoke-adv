from albumentations import ImageOnlyTransform
from albumentations import functional as F


class GaussianBlur(ImageOnlyTransform):
    def __init__(self, kernel_size=3, sigma=0, always_apply=False, p=0.5):
        super(GaussianBlur, self).__init__(always_apply, p)
        self.kernel_size = kernel_size
        self.sigma = sigma

    def apply(self, image, kernel_size=3, sigma=0, **params):
        return F.gaussian_blur(image, kernel_size, sigma=sigma)

    def get_params(self):
        return {"kernel_size": self.kernel_size, "sigma": self.sigma}

    def get_transform_init_args_names(self):
        return "kernel_size", "sigma"
