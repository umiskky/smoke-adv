import torch


class EarlyStop:
    def __init__(self, max_step=8, eps=0.0001) -> None:
        self._count = 0
        self._eps = eps
        self._max_step = max_step
        self._last_min = None

    def step(self, inspect) -> bool:
        if isinstance(inspect, torch.Tensor):
            with torch.no_grad():
                assert len(inspect.shape) == 1
                inspect = inspect.clone().cpu().item()
        if self._last_min is None:
            self._count = 0
            self._last_min = inspect
            return True
        delta = self._last_min - inspect
        if delta < self._eps:
            self._count += 1
        else:
            self._count = 0
            self._last_min = inspect
        if self._count > self._max_step:
            return False
        return True


