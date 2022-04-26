import torch
import torch.nn.functional as F


class PGDOptimizer:
    def __init__(self, params: dict, alpha: float = 0.001,
                 clip_min: float = 0.01, clip_max: float = 0.1,
                 device="cpu"):
        if clip_max <= clip_min:
            raise ValueError("Invalid clip: min={}, max={}".format(clip_min, clip_max))
        if alpha < 0.0:
            raise ValueError("Invalid iter eps value: {}".format(alpha))
        self._super_params = dict(alpha=alpha, clip_min=clip_min, clip_max=clip_max)
        self._params = params
        self._grad_container = []
        self._device = device

    @property
    def super_params(self):
        return self._super_params

    @torch.no_grad()
    def record(self):
        param = self._params.get("adv_texture_hls")
        if param is not None:
            if param.grad is None:
                self._grad_container.append(0)
                param.requires_grad_(False)
            else:
                d_param = param.grad
                # self._grad_container.append(d_param.clone().cpu())
                self._grad_container.append(d_param.clone())
                param.requires_grad_(False)

    @torch.no_grad()
    def step(self, step_type="mean", step_loss_list: list = None):
        """step_type can be mean or softmax"""
        # update params each epoch
        d_param = None
        alpha = self._super_params.get("alpha")
        clip_min = self._super_params.get("clip_min")
        clip_max = self._super_params.get("clip_max")

        if "softmax" == step_type and step_loss_list is not None:
            d_param = self._softmax(self._grad_container, step_loss_list)
        elif "mean" == step_type:
            # calculate in cpu to avoid out of memory error
            d_param = self._mean(self._grad_container)

        adv_texture_hls = self._params.get("adv_texture_hls")
        ori_texture_hls = self._params.get("ori_texture_hls")
        mask = self._params.get("mask")
        # PGD
        new_adv_texture_hls = adv_texture_hls + alpha * mask * torch.sign(d_param.to(adv_texture_hls.device))
        eta = torch.clamp(new_adv_texture_hls - ori_texture_hls, min=clip_min, max=clip_max)
        adv_texture_hls[:] = ori_texture_hls + mask * eta
        # clear
        self._grad_container = []

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
            grad_sum += weight[grad_idx].to(grad.device) * grad
        return grad_sum
