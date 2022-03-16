import torch

if __name__ == "__main__":
    test = torch.tensor([[[1, 2], [3, 4], [5, 12]], [[8, 7], [10, 9], [11, 6]]])

    print(torch.nonzero(test))

