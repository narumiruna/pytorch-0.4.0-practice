import argparse

import torch
import torch.nn.functional as F
from torch import nn, optim

from loader import mnist_loader
from model import Net
from utils import AverageMeter, AccuracyMeter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}.'.format(device))

    train_loader, valid_loader = mnist_loader(args.root, args.batch_size)
    model = Net().to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    def train():
        model.train()

        train_loss = AverageMeter()
        train_acc = AccuracyMeter()
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            loss = F.cross_entropy(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = output.data.max(dim=1)[1]
            correct = int(pred.eq(y.data).cpu().sum())

            train_loss.update(float(loss.data), number=x.size(0))
            train_acc.update(correct, number=x.size(0))

        return train_loss.average, train_acc.accuracy

    def validate():
        model.eval()

        valid_loss = AverageMeter()
        valid_acc = AccuracyMeter()
        with torch.no_grad():
            for i, (x, y) in enumerate(valid_loader):
                x = x.to(device)
                y = y.to(device)

                output = model(x)
                loss = F.cross_entropy(output, y)

                pred = output.data.max(dim=1)[1]
                correct = int(pred.eq(y.data).cpu().sum())

                valid_loss.update(float(loss.data), number=x.size(0))
                valid_acc.update(correct, number=x.size(0))

        return valid_loss.average, valid_acc.accuracy

    for epoch in range(args.epochs):
        train_loss, train_acc = train()
        valid_loss, valid_acc = validate()

        print('Train epoch: {}/{},'.format(epoch + 1, args.epochs),
              'train loss: {:.6f}, train acc: {:.2f}%,'.format(train_loss, train_acc * 100),
              'valid loss: {:.6f}, valid acc: {:.2f}%.'.format(valid_loss, valid_acc * 100))


if __name__ == '__main__':
    main()
