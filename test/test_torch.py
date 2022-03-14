import torch

if __name__ == "__main__":
    device = torch.device("cuda:1")
    model = torch.load("/home/dingxl/workspace/smoke-adv/data/results/2022-03-13-10-15/texture/00005_002.57271_patch.pth")
    pass
