import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import mlflow
import mlflow.pytorch

def train():
    mlflow.start_run()
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = datasets.FakeData(transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(2):  # minimal training for demo
        for x, y in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    mlflow.pytorch.log_model(model, "model")
    mlflow.end_run()

if __name__ == "__main__":
    train()
