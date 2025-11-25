"""
Author: Marc DeCarlo, Suman Kumar
Email: marcadecarlo@gmail.com
Date: November 25, 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import snntorch as snn
from snntorch import spikegen

# this should auto-patch:
#   - snn.TernaryLeaky
#   - spikegen.ternary_rate
#   - surrogate.atan_ternary 
import snntorch_ternary


# -----------------------------
# Hyperparameters
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 64
num_epochs = 2        
num_steps = 5         
beta = 0.9            
learning_rate = 1e-3


# -----------------------------
# Datasets & Dataloaders
# -----------------------------
# 1) ToTensor -> [0, 1]
# 2) Lambda: map to [-1, 1] so negative spikes exist
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2.0 - 1.0),  # [-1, 1]
])

train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True,
)

test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True,
)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
)


# -----------------------------
# Ternary SNN Model
# -----------------------------
class TernaryMNISTNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(28 * 28, 256)
        self.lif1 = snn.TernaryLeaky(beta=beta)

        self.fc2 = nn.Linear(256, 10)
        self.lif2 = snn.TernaryLeaky(beta=beta, output=True)

    def forward(self, x):
        """
        x: [T, batch, 784] ternary spikes in {-1,0,+1}
        returns: firing rates over time, shape [batch, 10]
        """
        batch_size = x.size(1)
        mem1 = torch.zeros(batch_size, 256, device=x.device)
        mem2 = torch.zeros(batch_size, 10, device=x.device)

        spk2_rec = []

        for t in range(x.size(0)):
            cur1 = self.fc1(x[t])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)

        spk2_stack = torch.stack(spk2_rec, dim=0)
        out_fr = spk2_stack.mean(dim=0)  
        return out_fr


model = TernaryMNISTNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# -----------------------------
# Ternary encoding helper
# -----------------------------
def encode_ternary(images):
    """
    images: [batch, 1, 28, 28] in [-1, 1]
    returns: ternary spikes [T, batch, 784]
    """
    batch = images.size(0)
    # flatten to [batch, 784]
    images_flat = images.view(batch, -1)
    # ternary_rate will interpret sign as spike sign and magnitude as probability
    spk = spikegen.ternary_rate(
        images_flat,
        num_steps=num_steps,
        gain=1.0,
        offset=0.0,
        first_spike_time=0,
        time_var_input=False,
    )
    return spk


# -----------------------------
# Training & Evaluation Loops
# -----------------------------
def train_one_epoch(epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)

        spk_in = encode_ternary(images)

        optimizer.zero_grad()
        outputs = model(spk_in)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

        if (batch_idx + 1) % 100 == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] "
                f"Step [{batch_idx+1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f} "
                f"Acc: {100.0 * correct / total:.2f}%"
            )

    avg_loss = total_loss / total
    avg_acc = 100.0 * correct / total
    print(f"==> Epoch {epoch}: Train Loss {avg_loss:.4f}, Acc {avg_acc:.2f}%")
    return avg_loss, avg_acc


def evaluate():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)

            spk_in = encode_ternary(images)
            outputs = model(spk_in)

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    acc = 100.0 * correct / total
    print(f"==> Test Accuracy: {acc:.2f}%")
    return acc


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("Using device:", device)
    print("Has TernaryLeaky?", hasattr(snn, "TernaryLeaky"))
    print("Has ternary_rate?", hasattr(spikegen, "ternary_rate"))

    for epoch in range(1, num_epochs + 1):
        train_one_epoch(epoch)
        evaluate()
