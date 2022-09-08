import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import shutil
shutil.rmtree('runs', True)

#hpu specific
from habana_frameworks.torch.utils.library_loader import load_habana_module
load_habana_module()
habana_device = torch.device("hpu")

#general
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
model = NeuralNetwork().to(habana_device)
model.train()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
train_dataloader = DataLoader(training_data, batch_size=64)
activities = []
activities.append(torch.profiler.ProfilerActivity.CPU)

#hpu specific
activities.append(torch.profiler.ProfilerActivity.HPU)

#general
with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=0, warmup=20, active=5, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./runs/fashion_mnist_experiment_1/'),
        record_shapes=True,
        with_stack=True) as prof:
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(habana_device), y.to(habana_device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        prof.step()