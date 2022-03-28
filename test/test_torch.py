import torch

if __name__ == "__main__":
    test = torch.tensor([[[1.0, 2], [3, 4]], [[8, 7], [10, 9]], [[5, 12], [11, 6]]], device="cuda:0")
    print((torch.norm(test, p='fro', dim=[1, 2])).split(1, dim=0))
    print(torch.norm(test, p=float('inf'), dim=[1, 2]))
    print(torch.norm(test, p=1, dim=[1, 2]))
    print(test)



