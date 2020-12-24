import torch
import sys
import pandas as pd
from torch import nn


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


if __name__ == "__main__":
    test_file = sys.argv[1]
    param = sys.argv[2]

    print("Start loading...")
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

    net.load_state_dict(torch.load(param))
    net.to(try_gpu())
    net.eval()

    test = pd.read_csv(test_file)

    print("Start predicting...")
    result = []
    for i in range(0, len(test)):
        map = torch.tensor([float(j) for j in test.iloc[i, 0].split(" ")]).reshape(48, 48).unsqueeze(0).unsqueeze(0).to(
            try_gpu())
        arr = net(map)
        result.append(int((torch.max(arr, dim=-1)).indices.cpu()))

    index = [i for i in range(1, len(test) + 1)]
    output = pd.DataFrame({'ID': index, 'emotion': result})
    output.to_csv('Answer.csv', index=None, encoding='utf8')