import torch
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils import data
from torch import nn
import time
from IPython import display
import seaborn as sns
import tensorflow as tf
import numpy as np
import random
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries


# load and preprocess data
source = pd.read_csv("../input/challenges-in-representation-learning-facial-expression-recognition-challenge/train.csv")
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


# construct net
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

net = nn.Sequential(stage1, stage2, stage3, stage4, stage5, nn.Flatten(), nn.Linear(256, 1024), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(1024, 7))

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


train_plot(net, train_iter, valid_iter, train_epochs, torch.optim.Adam(net.parameters(), lr=train_lr), False)
plt.savefig("Result.jpg")
plt.show()
torch.save(net.state_dict(), "VGG.param")

# fine-tune
net.load_state_dict(torch.load("./Net.param"))
train_plot(net, train_iter, valid_iter, tune_epochs, torch.optim.SGD(net.parameters(), lr=tune_lr), True)

# predict
net.load_state_dict(torch.load("./Net (fine-tune).param"))
net.eval()

test = pd.read_csv("../input/challenges-in-representation-learning-facial-expression-recognition-challenge/test.csv")

result = []
for i in range(0, len(test)):
    map = torch.tensor([float(j) for j in test.iloc[i, 0].split(" ")]).reshape(48 ,48).unsqueeze(0).unsqueeze(0).to(try_gpu())
    arr = net(map)
    result.append(int((torch.max(arr, dim=-1)).indices.cpu()))

index = [i for i in range(1, len(test) + 1)]
output = pd.DataFrame({'ID': index, 'emotion': result})
output.to_csv('Answer.csv', index = None, encoding = 'utf8')

# saliency map
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}


def compute_saliency_map(series, model):
    fig = torch.unsqueeze(processed_data[series][0], dim=0)
    model.eval()
    X_var = torch.autograd.Variable(fig, requires_grad=True)
    scores = model(X_var.to(try_gpu()))[0]
    scores.backward(torch.FloatTensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).to(try_gpu()))
    saliency_map = X_var.grad.data
    saliency_map = saliency_map.abs()
    saliency_map, i = torch.max(saliency_map, dim=1)
    saliency_map = saliency_map.squeeze()

    return saliency_map


def plot_saliency_maps(true_type, predict_type, num=5):
    for i in range(0, num):
        trial = random.randint(0, len(processed_data))
        pred = net(processed_data[trial][0].unsqueeze(0).to(try_gpu()))
        pred = int((torch.max(pred, dim=-1)).indices.cpu())
        while (pred != predict_type or processed_data[trial][1] != true_type):
            trial = random.randint(0, len(processed_data))
            pred = net(processed_data[trial][0].unsqueeze(0).to(try_gpu()))
            pred = int((torch.max(pred, dim=-1)).indices.cpu())
        plt.subplot(2, num, i + 1)
        plt.imshow(processed_data[trial][0][0], cmap="gray")
        plt.axis('off')
        plt.subplot(2, num, num + i + 1)
        saliency = compute_saliency_map(trial, net)
        plt.imshow(saliency)
        plt.axis('off')
        plt.gcf().set_size_inches(12, 5)
    plt.savefig(emotion_dict[true_type] + "(" + emotion_dict[predict_type] + ")")
    plt.show()


for i in range(0, 7):
    plot_saliency_maps(i, i)


# gradient ascent
def part_calculate(net, stage, layer, channel, matrix):
    ret = matrix
    stage = stage - 1
    for i in range(0, stage):
        ret = net[i](ret)
    if layer >= 1:
        ret = net[stage][0](ret)
    if layer >= 2:
        ret = net[stage][1](ret)
        ret = net[stage][2](ret)
        ret = net[stage][3](ret)
    if layer >= 3:
        ret = net[stage][4](ret)
        ret = net[stage][5](ret)
        ret = net[stage][6](ret)
    if layer >= 4:
        ret = net[stage][7](ret)
        ret = net[stage][8](ret)
        ret = net[stage][9](ret)
    return ret[0][channel-1]


def gradient_ascent(net, pic, lr, decay, epochs, stage, layer, channel):
    for round in range(0, epochs):
        loss = part_calculate(net, stage, layer, channel, pic).sum() * (-1)
        loss.backward()
        with torch.no_grad():
            pic.data.sub_(lr * pic.grad)
            for i in range(0, 48):
                for j in range(0, 48):
                    if pic[0][0][i][j] < 0:
                        pic[0][0][i][j] = 0
                    elif pic[0][0][i][j] > 255:
                        pic[0][0][i][j] = 255
        pic.grad.zero_()
        lr = lr * decay
    return pic


def find_best_activation(net, stage, layer, channel):
    X = torch.rand(size=(1, 1, 48, 48)).to(device='cuda') * 255
    X.requires_grad_(True)
    best = gradient_ascent(net, X, 100, 0.99, 300, stage, layer, channel)
    plt.imshow(best.cpu().detach().numpy()[0][0], cmap="gray")
    plt.savefig("stage" + str(stage) + " layer" + str(layer) + " channel" + str(channel) + ".jpg")
    plt.show()


for i in range(1, 6):
    for j in range(1, 5):
        find_best_activation(net, i, j, 1)


# filter output
def plot_intermediate_layer(net, stage, layer, channel):
    img = part_calculate(net, stage, layer, channel, processed_data[9][0].unsqueeze(0).cuda())
    plt.imshow(img.cpu().detach().numpy(), cmap="gray")
    plt.savefig("stage" + str(stage) + " layer" + str(layer) + " channel" + str(channel) + ".jpg")
    plt.show()


for i in range(1, 6):
    for j in range(1, 5):
        plot_intermediate_layer(net, i, j, 1)

# confusion matrix
y_pred = []
y_true = []
for X, y in valid_iter:
    X = X.to(try_gpu())
    ans = net(X)
    for i in range(0, len(ans)):
        temp = list(ans[i])
        y_pred.append(temp.index(max(temp)))
        y_true.append(int(y[i]))

confusion_matrix = tf.math.confusion_matrix(labels=y_true, predictions=y_pred).numpy()
confusion_matrix = np.around(confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis], decimals=2)
confusion_matrix = pd.DataFrame(confusion_matrix, index=emotion_dict.values(), columns=emotion_dict.values())

figure = plt.figure(figsize=(8, 8))
sns.heatmap(confusion_matrix, annot=True, cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.savefig("Confusion matrix.jpg")
plt.show()


# lime
def predict(images):
    trans = transforms.Compose([transforms.ToTensor()])
    batch = torch.stack(tuple(trans(i) for i in images), dim=0)
    gray_pic = torch.zeros(size=(batch.size()[0], 1, 48, 48), device='cuda')
    for i in range(0, batch.size()[0]):
        gray_pic[i] = batch[i][0].unsqueeze(0)
    gray_pic.cuda()
    logits = net(gray_pic)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()


def analyse(series):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(processed_data[series][0][0], predict, top_labels=7, num_samples=1000)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, hide_rest=False)
    img_boundry = mark_boundaries(temp/255, mask)
    return img_boundry


def plot_lime_maps(true_type, predict_type, num=5):
    for i in range(0, num):
        trial = random.randint(0, len(processed_data))
        pred = net(processed_data[trial][0].unsqueeze(0).to(device='cuda'))
        pred = int((torch.max(pred, dim=-1)).indices.cpu())
        while (pred != predict_type or processed_data[trial][1] != true_type):
            trial = random.randint(0, len(processed_data))
            pred = net(processed_data[trial][0].unsqueeze(0).to(device='cuda'))
            pred = int((torch.max(pred, dim=-1)).indices.cpu())
        plt.subplot(2, num, i + 1)
        plt.imshow(processed_data[trial][0][0], cmap="gray")
        plt.axis('off')
        plt.subplot(2, num, num + i + 1)
        boundary = analyse(trial)
        plt.imshow(boundary)
        plt.axis('off')
        plt.gcf().set_size_inches(12, 5)
    plt.savefig(emotion_dict[true_type] + "(" + emotion_dict[predict_type] + ")")
    plt.show()


for i in range(0, 7):
    plot_lime_maps(i, i)
plot_lime_maps(1, 0)
plot_lime_maps(2, 4)
plot_lime_maps(2, 5)
plot_lime_maps(6, 4)
plot_lime_maps(4, 6)


# filter analyse
def filter_analyse(series, num=5):
    image = processed_data[series][0][0].unsqueeze(0).unsqueeze(0).cuda()
    for i in range(1, 6):
        for j in range(1, 5):
            for k in range(0, num):
                sample = random.randint(1, 64)
                output = part_calculate(net, i, j, sample, image)
                plt.subplot(20, num, 20 * i + 5 * j + k - 24)
                plt.imshow(output.cpu().detach().numpy(), cmap="gray")
                plt.axis('off')
                plt.gcf().set_size_inches(10, 50)
                plt.savefig('Filter analyse')
    plt.show()


filter_analyse(16)
