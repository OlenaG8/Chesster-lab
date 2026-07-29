import torch
from torch.utils.data import DataLoader

from collect_dataset import get_dataset
from model import ChessCNN


dataset = get_dataset( "../dataset/raw")
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = ChessCNN()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(30):
    total_loss=0

    for images,labels in loader:
        output = model(images)
        loss = loss_fn(output,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(epoch, total_loss)

torch.save(model.state_dict(), "chess_piece_cnn.pt")