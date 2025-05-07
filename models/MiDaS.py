import torch
import urllib.request
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from PIL import Image

model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
model.eval()

transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

# Load image and preprocess
img = Image.open("example.jpg")
input_tensor = transform(img).unsqueeze(0)

# Predict
with torch.no_grad():
    depth = model(input_tensor)

# Resize to original resolution
depth = torch.nn.functional.interpolate(
    depth.unsqueeze(1),
    size=img.size[::-1],
    mode="bilinear",
    align_corners=False,
).squeeze()

depth_np = depth.cpu().numpy()


