import torch
import torch.nn as nn


test_input = torch.rand(1, 3, 28, 28)

model = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(7*7*128, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Linear(128, 10),
    nn.Softmax()
)
