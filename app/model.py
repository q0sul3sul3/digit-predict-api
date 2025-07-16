import os

import torch
from PIL import Image
from torchvision import transforms

from app.cnn_model import CNNTransformer

base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, 'model_weights.pth')
model = CNNTransformer()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# 預處理轉換
transform = transforms.Compose(
    [
        transforms.Resize((28, 28)),
        transforms.ToTensor(),  # 自動轉為 [1, 28, 28] 並除以 255
    ]
)


def predict_digit(image: Image.Image) -> int:
    image = image.convert('L')  # 轉為灰階
    img_tensor = transform(image).unsqueeze(0)  # [1, 1, 28, 28]
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1)
    return pred.item()
