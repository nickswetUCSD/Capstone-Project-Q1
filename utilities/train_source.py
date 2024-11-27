import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
import torch.utils.data as torch_data 
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import random_split
from torch.utils.data import TensorDataset, random_split
from robustbench.data import load_cifar10c
import copy
import json
import argparse
from collections import defaultdict

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load command line arguments
parser = argparse.ArgumentParser(description="Train a source classifier on CIFAR-10 with configurable losses")
parser.add_argument('--loss', type=str, default='cross_entropy', choices=['cross_entropy', 'poly'],
                    help='Loss function to use: "cross_entropy" or "poly"')

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, feature_maps=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out = self.layer4(out3)
        out = F.avg_pool2d(out, 4)
        out4 = out.view(out.size(0), -1)
        out = self.linear(out4)

        if feature_maps:
            return out, [out1, out2, out3, out4]
        else:
            return out

class Normalized_ResNet(ResNet):
    def __init__(self, device="cuda", depth=18, num_classes=10):
        if depth == 18:
            super(Normalized_ResNet, self).__init__(BasicBlock, [2,2,2,2], num_classes)
        elif depth == 50:
            super(Normalized_ResNet, self).__init__(Bottleneck, [3,4,6,3], num_classes)
        elif depth == 26:
            super(Normalized_ResNet, self).__init__(BasicBlock, [3,3,3,3], num_classes)
        else:
            pass 

        self.mu = torch.Tensor([0.4914, 0.4822, 0.4465]).float().view(3, 1, 1).to(device)
        self.sigma = torch.Tensor([0.2023, 0.1994, 0.2010]).float().view(3, 1, 1).to(device)

    def forward(self, x, feature_maps=False):
        x = (x - self.mu) / self.sigma
        return super(Normalized_ResNet, self).forward(x, feature_maps)
    
class PolyLoss(nn.Module):
    def __init__(self, epsilon=1.0):
        super(PolyLoss, self).__init__()
        self.epsilon = epsilon
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')  # Use un-reduced Cross-Entropy
    
    def forward(self, preds, labels):
        # Compute Cross-Entropy loss
        ce_loss = self.cross_entropy(preds, labels)
        
        # Apply softmax to compute probabilities
        probs = F.softmax(preds, dim=1)
        
        # Extract probabilities for the correct class
        correct_class_probs = probs[torch.arange(preds.size(0)), labels]
        
        # Compute polynomial term
        poly_term = self.epsilon * (1 - correct_class_probs)
        
        # Combine losses
        return ce_loss + poly_term

save_path = 'saved_models/pretrained/trained_model.pth'
net = Normalized_ResNet(depth=26)
net.to(device)
# net = torch.nn.DataParallel(net)
cudnn.benchmark = True

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

train_data = torchvision.datasets.CIFAR10("./data", True, transform=train_transform, download=True)
test_data = torchvision.datasets.CIFAR10("./data", False, transform=transforms.Compose([transforms.ToTensor()]), download=True)
subset_size = 25000
train_data, _ = random_split(train_data, [subset_size, len(train_data) - subset_size])
test_data, _ = random_split(test_data, [subset_size, len(test_data) - subset_size])

train_loader = torch_data.DataLoader(train_data, batch_size=256, shuffle=True, num_workers=4)
test_loader = torch_data.DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)

optimizer = torch.optim.SGD(net.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

if args.loss == 'cross_entropy':
    loss_function = nn.CrossEntropyLoss(reduction='none')
elif args.loss == 'poly':
    loss_function = PolyLoss()

best_acc = 0.0

train_loss_history = []
test_acc_history = []

for epoch in range(200):
    net.train()
    epoch_loss = 0.0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        preds = net(data)
        loss = loss_function(preds, labels).mean()
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(net.parameters(), 1, norm_type=2.0)
        optimizer.step()

    avg_epoch_loss = epoch_loss / len(train_loader)
    train_loss_history.append(avg_epoch_loss)    
    
    scheduler.step()

    acc = 0.0
    net.eval()
    for batch_idx, (data, labels) in enumerate(test_loader):
        data, labels = data.to(device), labels.to(device)
        preds = net(data)
        acc += (preds.max(1)[1] == labels).float().sum()

    acc = acc / 10000
    test_acc_history.append(acc.item())
    
    print(f"Epoch : {epoch} : Acc : {acc}")

    if acc > best_acc:
        best_acc = acc 
        torch.save({"net": net.state_dict()}, save_path)

with open('progress/test_acc_history.txt', 'w') as f:
    for number in test_acc_history:
        f.write(f'{number}\n')
with open('progress/train_loss_history.txt', 'w') as f:
    for number in train_loss_history:
        f.write(f'{number}\n')