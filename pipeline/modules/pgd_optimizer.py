import torch
import torch.nn.functional as F


class PGDOptimizer:
    def __init__(self, params: list, alpha: float = 0.001,
                 clip_min: float = 0.01, clip_max: float = 0.1,
                 position=None, size=None, device="cpu"):
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
        self._device = device

    @property
    def super_params(self):
        return self._super_params

    @torch.no_grad()
    def record(self):
        for param_idx, param in enumerate(self._params):
            if param.grad is None:
                self._grad_container[param_idx].append(0)
                param.requires_grad_(False)
            else:
                d_param = param.grad
                self._grad_container[param_idx].append(d_param.clone().cpu())
                param.requires_grad_(False)

    @torch.no_grad()
    def step(self, step_type="mean", step_loss_list: list = None):
        """step_type can be mean or softmax"""
        # update params each epoch
        alpha = self._super_params.get("alpha")
        clip_min = self._super_params.get("clip_min")
        clip_max = self._super_params.get("clip_max")
        x_l, y_l = self._super_params.get("position")
        size = self._super_params.get("size")
        for param_idx, param in enumerate(self._params):
            if len(self._grad_container[param_idx]) == 0:
                continue
            if "softmax" == step_type and step_loss_list is not None:
                assert isinstance(step_loss_list, list) and len(step_loss_list) == len(self._grad_container[param_idx])
                d_param = self._softmax(self._grad_container[param_idx], step_loss_list)
            else:
                # calculate in cpu to avoid out of memory error
                d_param = self._mean(self._grad_container[param_idx])

            perturbation = alpha * torch.sign(d_param)
            perturbation = perturbation.to(self._device)
            # 1 -> l channel
            param[:, 1, y_l: y_l + size[0], x_l: x_l + size[1]] += \
                perturbation[:, 1, y_l: y_l + size[0], x_l: x_l + size[1]]
            param[:, 1, y_l: y_l + size[0], x_l: x_l + size[1]] = \
                torch.clamp(param[:, 1, y_l: y_l + size[0], x_l: x_l + size[1]], min=clip_min, max=clip_max)
        # clear
        self._grad_container = [[] * len(self._params)]

    @staticmethod
    def _mean(grad_list: list):
        grad_sum = None
        for grad in grad_list:
            if grad_sum is None:
                grad_sum = grad
            else:
                grad_sum += grad
        return grad_sum / len(grad_list)

    @staticmethod
    def _softmax(grad_list: list, weight_list: list):
        """Using softmax function to weight grad"""
        grad_sum = 0
        weight = torch.flatten(F.softmax(torch.tensor(weight_list), dim=0))
        assert len(grad_list) == weight.shape[0]
        for grad_idx, grad in enumerate(grad_list):
            grad_sum += weight[grad_idx] * grad
        return grad_sum
