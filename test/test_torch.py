import os.path as osp
import random

import torch


def patch(patch_path):
    project_path = "/home/dingxl/workspace/smoke-adv/"
    fp = osp.join(project_path, patch_path)
    if osp.exists(fp):
        state = torch.load(fp)
        patch = state.get("patch")
        pass


if __name__ == "__main__":
    # patch_path = "data/results/2022-03-28-17-07/visualization/patch/00270_000.49891_patch.pth"
    # patch(patch_path)
    times = 180
    angle_range = [90, 270]
    angle_list = [random.randint(0, (angle_range[1] - angle_range[0]) // times) +
                  i * ((angle_range[1] - angle_range[0]) // times) +
                  angle_range[0]
                  for i in range(times)]
    print(angle_list)

