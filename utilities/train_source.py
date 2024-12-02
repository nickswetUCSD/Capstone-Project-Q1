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
import os
from models import Normalized_ResNet
from collections import defaultdict

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load command line arguments
parser = argparse.ArgumentParser(description="Train a source classifier on CIFAR-10 with configurable losses")
parser.add_argument('--loss', type=str, default='cross_entropy', choices=['cross_entropy', 'poly'],
                    help='Loss function to use: "cross_entropy" or "poly"')
args = parser.parse_args()
    
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
directory = os.path.dirname(save_path)
if not os.path.exists(directory):
    os.makedirs(directory)
    print(f"Directory '{directory}' created.")
else:
    print(f"Directory '{directory}' already exists.")

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

for epoch in range(100):
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

test_acc_path = 'progress/test_acc_history.txt'
train_loss_path = 'progress/train_loss_history.txt'

# Create only the parent directory, not the full path
directory = os.path.dirname(test_acc_path)
if not os.path.exists(directory):
    os.makedirs(directory)
    print(f"Directory '{directory}' created.")
else:
    print(f"Directory '{directory}' already exists.")

# Write history to files
with open(test_acc_path, 'w') as f:
    for number in test_acc_history:
        f.write(f'{number}\n')
    print(f'Test accuracy history written to {test_acc_path}')

with open(train_loss_path, 'w') as f:
    for number in train_loss_history:
        f.write(f'{number}\n')
    print(f'Train loss history written to {train_loss_path}')