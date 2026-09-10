import torch
from torch.utils.data import DataLoader, random_split

from collect_dataset import get_dataset
from model import ChessCNN

#dataset = get_dataset("./dataset/raw")                  # CNN - V1
#dataset = get_dataset("./dataset/backgrounds_separated") # CNN - V2
#dataset = get_dataset("./dataset/binary")               # CNN - V3
dataset = get_dataset("./dataset/6-classes")             # CNN - V4

print("Classes:", dataset.class_to_idx)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = ChessCNN()
best_val_acc = 0

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(30):

    # TRAIN
    model.train()

    train_correct = 0
    train_total = 0
    train_loss = 0

    for images, labels in train_loader:
        output = model(images)
        loss = loss_fn(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        predictions = output.argmax(1)
        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)

    train_acc = train_correct / train_total

    # VALIDATION
    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            output = model(images)
            predictions = output.argmax(1)

            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total

    print(
        f"Epoch {epoch + 1:02d} | "
        f"loss={train_loss:.3f} | "
        f"train_acc={train_acc:.3f} | "
        f"val_acc={val_acc:.3f}"
    )
    if val_acc > best_val_acc:
        best_val_acc = val_acc

        torch.save(model.state_dict(),"chess_piece_cnn_v4.pt")

        print("Saved best model!")
