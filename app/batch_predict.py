import csv
import os

import torch
from PIL import Image
from torchvision import transforms

from app.cnn_model import CNNTransformer

model = CNNTransformer()
model.load_state_dict(torch.load('model_weights.pth', map_location=torch.device('cpu')))
model.eval()

# 圖片預處理
transform = transforms.Compose(
    [
        transforms.Resize((28, 28)),
        transforms.ToTensor(),  # [1, 28, 28]
    ]
)


# 預測函數
def predict_image(image_path):
    image = Image.open(image_path).convert('L')  # 灰階
    image = transform(image).unsqueeze(0)  # [1, 1, 28, 28]
    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1)
    return pred.item()


# 資料夾路徑
test_dir = 'test'
image_files = [
    f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))
]

# 寫入 CSV
with open('result.csv', mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'prediction'])
    for filename in image_files:
        path = os.path.join(test_dir, filename)
        prediction = predict_image(path)
        writer.writerow([filename, prediction])

print('預測完成！結果已寫入 result.csv')
