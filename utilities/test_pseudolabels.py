import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset, DataLoader
from robustbench.data import load_cifar10c
import copy
import json
import argparse
from models import Normalized_ResNet
from collections import defaultdict

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Test-Time Adaptation with Pseudo-Labels")
parser.add_argument('--pseudo_label', type=str, choices=['hard', 'conjugate'], required=True,
                    help="Type of pseudo-labeling to use: 'hard' or 'conjugate'")
args = parser.parse_args()

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load pretrained model
net = Normalized_ResNet(depth=26)
checkpoint = torch.load("saved_models/pretrained/trained_model.pth", map_location=device)  # Replace with the actual checkpoint path
net.load_state_dict(checkpoint['net'])
net.to(device)
net.eval()

# Hyperparameters and configurations
subset_size = 5000  # Reduce for testing
batch_size = 200
severity_list = [5, 4, 3, 2, 1]
corruption_type_list = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression"
]
eps = 1e-3  # Regularization factor for conjugate pseudo-labels
error_dict = defaultdict(dict)


def tta_loss_fn(outputs, softmax_prob, pseudo_label_type, num_classes):
    """Compute the TTA loss based on the pseudo-labeling method."""
    if pseudo_label_type == "hard":
        # Hard pseudo-labeling
        yp = outputs.max(1)[1]
        y_star = F.one_hot(yp, num_classes=num_classes)
        return torch.logsumexp(outputs, dim=1) - torch.sum(y_star * outputs, dim=1)
    
    elif pseudo_label_type == "conjugate":
        # Conjugate pseudo-labeling
        smax_inp = softmax_prob
        eye = torch.eye(num_classes).to(outputs.device).reshape((1, num_classes, num_classes)).repeat(outputs.shape[0], 1, 1)
        t2 = eps * torch.diag_embed(smax_inp)
        smax_inp = torch.unsqueeze(smax_inp, 2)
        t3 = eps * torch.bmm(smax_inp, torch.transpose(smax_inp, 1, 2))
        matrix = eye + t2 - t3
        y_star = torch.linalg.solve(matrix, smax_inp).squeeze()
        pseudo_prob = y_star
        return torch.logsumexp(outputs, dim=1) - (pseudo_prob * outputs - eps * pseudo_prob * (1 - softmax_prob)).sum(dim=1)


# Main evaluation loop
for severity in severity_list:
    for corruption_type in corruption_type_list:
        # Load corrupted CIFAR-10 data
        x_test, y_test = load_cifar10c(subset_size, severity, './data', True, [corruption_type])
        num_classes = 10
        test_dataset = TensorDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print(f"Evaluating corruption: {corruption_type}, severity: {severity}")
        net_test = copy.deepcopy(net)

        acc = 0.0
        step_acc_list = []  # To store accuracy after each gradient descent step

        for x_curr, y_curr in test_loader:
            x_curr, y_curr = x_curr.to(device), y_curr.to(device)
            x_curr.requires_grad = False

            # Perform test-time adaptation
            outputs = net_test(x_curr)
            outputs = outputs / 1.0  # Optional temperature scaling
            softmax_prob = F.softmax(outputs, dim=1)

            loss = tta_loss_fn(outputs, softmax_prob, args.pseudo_label, num_classes).mean()

            # Perform optimization step
            for param in net_test.parameters():
                param.grad = None
            loss.backward()
            for param in net_test.parameters():
                if param.grad is not None:
                    param.data -= 1e-3 * param.grad  # Learning rate for TTA adaptation

            # Compute accuracy after this step
            outputs_new = net_test(x_curr)
            step_accuracy = (outputs_new.max(1)[1] == y_curr).float().mean().item()
            step_acc_list.append(step_accuracy)

            # Update overall accuracy
            acc += (outputs_new.max(1)[1] == y_curr).float().sum()

        acc /= subset_size
        err = 1.0 - acc
        print(f"Error for {corruption_type} at severity {severity}: {err:.2%}")
        error_dict[corruption_type][severity] = {
            "error": err.item(),
            "step_accuracy": step_acc_list  # Store accuracy per step
        }

# Save results to JSON
with open("progress/error_rates_tta.json", "w") as f:
    json.dump(error_dict, f)

print("Evaluation complete. Results saved to progress/error_rates_tta.json")