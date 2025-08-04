import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim  # Needed for optimizer

# 1. Transform images to tensors
transform = transforms.ToTensor()

# 2. Load MNIST training data
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# 3. Define Neural Network
class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # FIXED: you had a typo "veiv"
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 4. Initialize Model
model = ImageClassifier()

# 5. Loss and Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Training Loop
for epoch in range(5):  # Use 5 epochs for now
    for images, labels in train_loader:
        preds = model(images)
        loss = loss_fn(preds, labels)

        optimizer.zero_grad()
        loss.backward()   # FIXED: it was `loss.backwards()`
        optimizer.step()

    print(f"Epoch {epoch+1} completed. Loss: {loss.item():.4f}")

    
