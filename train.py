import torch
import sys
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils import data
from torch import nn
import time
from IPython import display
import numpy as np
import random


# train
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)


class Timer:
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def sum(self):
        return sum(self.times)


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def __getitem__(self, idx):
        return self.data[idx]


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        display.set_matplotlib_formats('svg')
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.clear_output(wait=True)


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = argmax(y_hat, axis=1)
    cmp = astype(y_hat, y.dtype) == y
    return float(reduce_sum(astype(cmp, y.dtype)))


def evaluate_accuracy_gpu(net, data_iter, device=None):
    net.eval()
    if not device:
        device = next(iter(net.parameters())).device
    metric = Accumulator(2)
    for X, y in data_iter:
        X, y = X.to(device), y.to(device)
        metric.add(accuracy(net(X), y), size(y))
    return metric[0] / metric[1]


def train_plot(net, train_iter, valid_iter, num_epochs, opt, tune,
               device=try_gpu()):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)

    if tune == False:
        net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = opt
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', xlim=[0, num_epochs],
                        legend=['loss', 'train acc', 'valid acc'])
    timer = Timer()
    best_acc = 0

    for epoch in range(num_epochs):
        metric = Accumulator(3)
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            net.train()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]

            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))
        valid_acc = evaluate_accuracy_gpu(net, valid_iter)
        if valid_acc > best_acc:
            print("New best accuracy: {}".format(valid_acc))
            best_acc = valid_acc
            if tune == False:
                torch.save(net.state_dict(), "Net.param")
            else:
                torch.save(net.state_dict(), "Net (fine-tune).param")
        animator.add(epoch + 1, (None, None, valid_acc))
        print("Epoch: {}/{}, train accuray: {}, valid accuracy: {}".format(epoch, num_epochs, train_acc, valid_acc))
    print(f'loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'valid acc {valid_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
    if tune == False:
        plt.savefig("Result.jpg")
    else:
        plt.savefig("Result (fine-tune).jpg")
    plt.show()


if __name__ == "__main__":
    train_file = sys.argv[1]
    print("Start loading...")
    source = pd.read_csv(train_file)
    processed_data = []
    for i in range(0, len(source)):
        item = source.iloc[i, 1]
        temp1 = [float(j) for j in item.split(" ")]
        if not any(temp1):
            continue
        temp2 = []
        for j in range(0, 48):
            temp2.append(temp1[48 * j: 48 * j + 48])

        if sum(temp2, []).count(np.argmax(np.bincount(sum(temp2, [])))) >= 48 * 48 / 2:
            continue
        temp3 = [torch.tensor([temp2]), source.iloc[i, 0]]
        processed_data.append(temp3)

    train_lr, tune_lr, train_epochs, tune_epochs, batch_size = 0.001, 0.0001, 50, 50, 512
    random.shuffle(processed_data)
    train_iter = data.DataLoader(processed_data[:int(len(source) * 0.9)], batch_size, shuffle=True)
    valid_iter = data.DataLoader(processed_data[int(len(source) * 0.9):], batch_size, shuffle=False)
    print("Load success!")
    stage1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
                           nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
                           nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
                           nn.MaxPool2d(kernel_size=2), nn.Dropout(p=0.5))

    stage2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                           nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                           nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                           nn.MaxPool2d(kernel_size=2), nn.Dropout(p=0.5))

    stage3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                           nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                           nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                           nn.MaxPool2d(kernel_size=2), nn.Dropout(p=0.5))

    stage4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                           nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                           nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                           nn.MaxPool2d(kernel_size=2), nn.Dropout(p=0.5))

    stage5 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
                           nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
                           nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
                           nn.MaxPool2d(kernel_size=2), nn.Dropout(p=0.5))

    net = nn.Sequential(stage1, stage2, stage3, stage4, stage5, nn.Flatten(), nn.Linear(256, 1024), nn.ReLU(),
                        nn.Dropout(p=0.5), nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(1024, 7))

    # train
    print("Start training...")
    train_plot(net, train_iter, valid_iter, train_epochs, torch.optim.Adam(net.parameters(), lr=train_lr), False)
    plt.savefig("Result.jpg")
    plt.show()

    # fine tune
    print("Start fine tuning...")
    net.load_state_dict(torch.load("./Net.param"))
    train_plot(net, train_iter, valid_iter, tune_epochs, torch.optim.SGD(net.parameters(), lr=tune_lr), True)
