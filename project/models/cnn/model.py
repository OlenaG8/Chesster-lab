import torch.nn as nn

"""
image of one single cell -> feature extractor -> classifier -> output

Convolutional layer:
    in_chanel -> RGB has 3 channels
    out_channels -> output channels (filters)

ReLU -> activation function (adds non-linearity after convolutional layer | x < 0 ? 0 : x )
nn.BatchNorm2d -> batch normalization (normalizes the input of each layer to improve training stability and performance)
nn.MaxPool2d -> max pooling layer (reduces the spacial dimensions of input by taking the maximum value in a region)
nn.Flatten -> flatten layer (flattens the input tensor into a 1D vector while preserving the batch dimension)

nn.CrossEntropyLoss -> cross entropy loss function (muli-class classification tasks)
"""

class ChessCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 13)
        )

    def forward(self,x):
        # function that specifies how data flows through the layers of the network
        x = self.features(x)
        return self.classifier(x)