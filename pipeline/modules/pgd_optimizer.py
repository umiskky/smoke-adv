import torch


class PGDOptimizer:
    def __init__(self, params: list, alpha: float = 0.001,
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
        self._super_params = dict(alpha=alpha, clip_min=clip_min, clip_max=clip_max, position=position, size=size)
        self._params = params
        self._grad_container = [[] * len(self._params)]

    @torch.no_grad()
    def record(self):
        for param_idx, param in enumerate(self._params):
            if param.grad is None:
                continue
            d_param = param.grad
            self._grad_container[param_idx].append(d_param.clone())
            param.requires_grad_(False)

    @torch.no_grad()
    def step(self):
        # update params each epoch
        alpha = self._super_params.get("alpha")
        clip_min = self._super_params.get("clip_min")
        clip_max = self._super_params.get("clip_max")
        x_l, y_l = self._super_params.get("position")
        size = self._super_params.get("size")
        for param_idx, param in enumerate(self._params):
            if len(self._grad_container[param_idx]) == 0:
                continue
            d_param_mean = self._mean(self._grad_container[param_idx])
            if alpha != 0:
                perturbation = alpha * torch.sign(d_param_mean)
                param[:, 1, y_l: y_l + size[0], x_l: x_l + size[1]] += \
                    perturbation[:, 1, y_l: y_l + size[0], x_l: x_l + size[1]]
                param[:, 1, y_l: y_l + size[0], x_l: x_l + size[1]] = \
                    torch.clamp(param[:, 1, y_l: y_l + size[0], x_l: x_l + size[1]], min=clip_min, max=clip_max)
        # clear
        self._grad_container = [[] * len(self._params)]

    @staticmethod
    def _mean(grad_list: list):
        if len(grad_list) == 0:
            return None
        grad_sum = None
        for grad in grad_list:
            if grad_sum is None:
                grad_sum = grad
            else:
                grad_sum += grad
        return grad_sum / len(grad_list)
