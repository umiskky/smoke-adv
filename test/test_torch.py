import torch

if __name__ == "__main__":
    test = torch.tensor([[[-1.0, 2], [3, 4], [5, 12]], [[8, 7], [10, 9], [11, 6]]])
    print(test)
    torch.clamp_(test[:, 0, :], min=0.1, max=0.5)
    print(test)

