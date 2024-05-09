import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import tqdm
from matplotlib import pyplot as plt


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
                transforms.Resize((32, 32)),
                transforms.RandomChoice(transforms=[
                    transforms.TrivialAugmentWide(),
                    transforms.Lambda(lambda x: x)],
                    p=[0.95, 0.05]),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((32, 32)),
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
        self.fc1 = nn.Linear(32 * 32 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def train(net, criterion, optimizer):
    epoch_num = 100
    val_num = 1
    train_acc = []
    val_acc = []
    max_acc = 0

    for epoch in range(epoch_num):
        print("Epoch:", epoch + 1)
        correct = 0
        total = 0
        for data in tqdm.tqdm(train_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # print(images, labels)
            outputs = net(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct += (torch.max(outputs, 1)[1] == labels).sum().item()
            total += labels.size(0)
        
        print("Train Accuracy:", correct / total)
        train_acc.append(correct / total)

        if epoch % val_num == 0:
            acc = validation(net)
            val_acc.append(acc)
            print("Validation Accuracy:", acc)
            if acc > max_acc:
                max_acc = acc
                checkpoint = {
                    "net": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(checkpoint, "checkpoint_FNN.pth")
                print("Model Saved!")

    print('Finished Training!')

    plt.plot(range(1, epoch_num + 1), train_acc, label="Train")
    plt.plot(range(1, epoch_num + 1), val_acc, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("result_FNN.png")

def validation(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm.tqdm(val_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def test(net):
    with open("result_FNN.txt", "w") as f:
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
    train_set = CifarDataset("trainset")
    val_set = CifarDataset("validset")
    test_set = CifarDataset("testset")

    train_loader = DataLoader(dataset=train_set, batch_size=128, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=128, shuffle=False)
    test_loader = DataLoader(dataset=test_set, batch_size=128, shuffle=False)

    net = Net().to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.AdamW(net.parameters(), lr=0.001)

    train(net, criterion, optimizer)

    # test(net)

