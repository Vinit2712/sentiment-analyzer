import torch 
from torch import nn

from model import SentimentLSTM
from dataset import test_loader

VOCAB_SIZE = 20002
EMBEDDING_DIM = 64
HIDDEN_SIZE = 64
NUM_CLASSES = 2

model = SentimentLSTM(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    hidden_size=HIDDEN_SIZE,
    num_classes=NUM_CLASSES
)

model.load_state_dict(
    torch.load(
        "sentiment_lstm.pth",
        weights_only=True
    )
)

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for reviews,labels in test_loader:
        outputs = model(reviews)
        prediction = torch.argmax(outputs,dim=1)
        total += labels.size(0)
        correct += (prediction == labels).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")