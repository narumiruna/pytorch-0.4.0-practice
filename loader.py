from torchvision import datasets, transforms
from torch.utils import data


def mnist_loader(root, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.13066047740239478,), (0.3081078087569972,))
    ])

    train_loader = data.DataLoader(datasets.MNIST(root,
                                                  train=True,
                                                  transform=transform,
                                                  download=True),
                                   batch_size=batch_size,
                                   shuffle=True)

    valid_loader = data.DataLoader(datasets.MNIST(root,
                                                  train=False,
                                                  transform=transform,
                                                  download=True),
                                   batch_size=batch_size,
                                   shuffle=False)

    return train_loader, valid_loader
