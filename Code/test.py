import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import densenet201
from PIL import Image
import tqdm
from matplotlib import pyplot as plt
import argparse


class CifarDataset(Dataset):
    def __init__(self, mode):
        with open("Dataset/" + mode + ".txt", "r") as f:
            self.data = f.readlines()
            if mode == "trainset" or mode == "validset":
                self.label = [int(line.split(" ")[1]) for line in self.data]
                self.data = [line.split(" ")[0] for line in self.data]
            else:
                self.label = None
                self.data = [line.strip() for line in self.data]
        if mode == "trainset":
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomChoice(transforms=[
                    transforms.TrivialAugmentWide(),
                    transforms.Lambda(lambda x: x)],
                    p=[0.95, 0.05]),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        self.data = [self.transform(Image.open("Dataset/image/" + data)) for data in self.data]

    def __getitem__(self, index):
        if self.label is not None:
            return self.data[index], self.label[index]
        else:
            return self.data[index], -1

    def __len__(self):
        return len(self.data)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.size = 70
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=1)
        self.size = (self.size - 11 + 2 * 1) // 2 + 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.size = (self.size - 3) // 2 + 1
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.size = (self.size - 5 + 2 * 2) // 1 + 1
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.size = (self.size - 3) // 2 + 1
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.size = (self.size - 3 + 2 * 1) // 1 + 1
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.size = (self.size - 3 + 2 * 1) // 1 + 1
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.size = (self.size - 3 + 2 * 1) // 1 + 1
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.size = (self.size - 3) // 2 + 1
        self.fc1 = nn.Linear(256 * self.size * self.size, 4096)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.maxpool3(x)
        x = x.view(-1, 256 * self.size * self.size)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


def test(net):
    with open(args.output, "w") as f:
        for data in tqdm.tqdm(test_loader):
            images, labels = data
            images = images.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            for label in predicted:
                f.write(str(label.item()) + "\n")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    test_set = CifarDataset("testset")

    test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=False)

    net = Net().to(device)

    arg = argparse.ArgumentParser()

    arg.add_argument("--model", type=str, required=True)

    arg.add_argument("--output", type=str, required=True)

    args = arg.parse_args()

    checkpoint = torch.load(args.model)

    net.load_state_dict(checkpoint["net"])

    test(net)

