import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import os

# ======================
# CONFIG
# ======================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# TRANSFORMS
# ======================
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ======================
# PATH FIX (AUTO)
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

train_path = os.path.join(BASE_DIR, "..", "fer2013", "train")
val_path   = os.path.join(BASE_DIR, "..", "fer2013", "test")

print("Train path:", train_path)
print("Val path:", val_path)

# ======================
# DATA
# ======================
train_ds = datasets.ImageFolder(train_path, transform=train_tf)
val_ds   = datasets.ImageFolder(val_path, transform=val_tf)

print("Classes:", train_ds.classes)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ======================
# MODEL
# ======================
model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)

# Freeze backbone (faster + stable)
for param in model.features.parameters():
    param.requires_grad = False

# Replace classifier (4 classes)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)

model = model.to(DEVICE)

# ======================
# LOSS & OPTIMIZER
# ======================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ======================
# TRAIN LOOP
# ======================
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 50 == 0:
            print(f"Batch {i} Loss: {loss.item():.4f}")

    print(f"Epoch Loss: {running_loss:.4f}")

# ======================
# SAVE MODEL
# ======================
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "emotion_model.pth")
torch.save(model.state_dict(), MODEL_SAVE_PATH)

print(f"✅ Model saved at: {MODEL_SAVE_PATH}")