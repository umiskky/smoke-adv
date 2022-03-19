import torch
import torch.optim as top


class PGDOptimizer(top.Optimizer):
    def __init__(self, params, alpha: float = 0.001,
                 clip_min: float = 0.01, clip_max: float = 0.1,
                 position=None, size=None):
        if position is None:
            position = [0, 0]
        if size is None:
            size = [1, 1]
        if clip_max <= clip_min:
            raise ValueError("Invalid clip: min={}, max={}".format(clip_min, clip_max))
        if alpha < 0.0:
            raise ValueError("Invalid iter eps value: {}".format(alpha))

        defaults = dict(alpha=alpha, clip_min=clip_min, clip_max=clip_max, position=position, size=size)
        super(PGDOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            alpha = group["alpha"]
            clip_min = group["clip_min"]
            clip_max = group["clip_max"]
            x_l, y_l = group["position"]
            size = group["size"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                d_param = param.grad
                if alpha != 0:
                    perturbation = alpha * torch.sign(d_param)
                    param[:, 1, y_l: y_l + size[0], x_l: x_l + size[1]] += \
                        perturbation[:, 1, y_l: y_l + size[0], x_l: x_l + size[1]]
                    param[:, 1, y_l: y_l + size[0], x_l: x_l + size[1]] = \
                        torch.clamp(param[:, 1, y_l: y_l + size[0], x_l: x_l + size[1]], min=clip_min, max=clip_max)
                param.requires_grad_(False)
                pass
