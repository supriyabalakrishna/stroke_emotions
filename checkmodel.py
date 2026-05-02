import torch
from torchvision.models import mobilenet_v3_small
import torch.nn as nn

# Load model
model = mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)

state_dict = torch.load("model/emotion_model.pth", map_location="cpu")
model.load_state_dict(state_dict)

model.eval()

print("✅ Model loaded successfully")

# Check output shape
dummy = torch.randn(1, 3, 224, 224)
output = model(dummy)

print("Output shape:", output.shape)