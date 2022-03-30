import albumentations as A


class Transform:
    def __init__(self, args: dict) -> None:
        self._type = args["type"]
        self._transform = None
        if "BitDepth" == self._type:
            r_bits = self._to_list(args["bit_depth"]["r_bits"])
            g_bits = self._to_list(args["bit_depth"]["g_bits"])
            b_bits = self._to_list(args["bit_depth"]["b_bits"])
            self._transform = A.Compose([
                A.Posterize(num_bits=[r_bits, g_bits, b_bits], always_apply=True)
            ])
        elif "MedianBlur" == self._type:
            # TODO
            blur_limit = args["median_blur"]["blur_limit"]
            self._transform = A.Compose([
                A.MedianBlur(blur_limit=blur_limit, always_apply=True)
            ])
            pass
        elif "GaussianBlur" == self._type:
            # TODO
            blur_limit = self._to_list(args["gaussian_blur"]["blur_limit"])
            sigma_limit = args["gaussian_blur"]["sigma_limit"]
            self._transform = A.Compose([
                A.GaussianBlur(blur_limit=blur_limit, sigma_limit=sigma_limit, always_apply=True)
            ])
            pass
        elif "JpegCompression" == self._type:
            quality_lower = args["jpeg_compression"]["quality_lower"]
            quality_upper = args["jpeg_compression"]["quality_upper"]
            self._transform = A.Compose([
                A.ImageCompression(quality_lower=quality_lower, quality_upper=quality_upper,
                                   compression_type=0, always_apply=True)
            ])

    def forward(self, image):
        pass

    @staticmethod
    def _to_list(item):
        if isinstance(item, list):
            return item
        return [item, item]