import os
import torch
import torchvision.transforms as transforms

class ChessCNN:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model {model_path} not found')

        size = os.path.getsize(model_path)
        if size == 0:
            raise RuntimeError(f'Model {model_path} is empty')

        self.model = torch.load(model_path, map_location="cpu")
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64,64)),
            transforms.ToTensor(),
        ])

    def predict(self, square):
        x = self.transform(square)
        x = x.unsqueeze(0)

        with torch.no_grad():
            output = self.model(x)

        cls = output.argmax(1)
        return cls.item()
