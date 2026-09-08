import os
import torch
import torchvision.transforms as transforms

from project.models.cnn.model import ChessCNN

class ChessCNNClassifier:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {model_path} not found")

        if os.path.getsize(model_path) == 0:
            raise RuntimeError(f"Model {model_path} is empty")

        self.model = ChessCNN()
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)

        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

    def predict(self, square):
        x = self.transform(square)
        x = x.unsqueeze(0)

        with torch.no_grad():
            output = self.model(x)

        cls = output.argmax(1)

        return cls.item()