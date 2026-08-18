import torch
from torch import nn,optim
from model import SentimentLSTM
from dataset import train_loader

VOCAB_SIZE = 20002
EMBEDDING_DIM = 64
HIDDEN_SIZE = 64
NUM_CLASSES = 2

LEARNING_RATE = 0.001
EPOCHS = 5

model = SentimentLSTM(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    hidden_size=HIDDEN_SIZE,
    num_classes=NUM_CLASSES   
)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr = LEARNING_RATE
)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for reviews , labels in train_loader:
        output = model(reviews)
        loss = criterion(output,labels)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss/len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}]")
    print(f"Loss: {average_loss:.4f}")

torch.save(model.state_dict(), "sentiment_lstm.pth")