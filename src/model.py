import torch
from torch import nn

class SentimentLSTM(nn.Module):

    def __init__(
            self,
            vocab_size,
            embedding_dim,
            hidden_size,
            num_classes
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = embedding_dim,
            padding_idx = 0
        )

        self.lstm = nn.LSTM(
            input_size = embedding_dim,
            hidden_size = hidden_size,
            batch_first = True
        )

        self.fc = nn.Linear(
            in_features = hidden_size,
            out_features = num_classes
        )

    def forward(self,x):

        x = self.embedding(x)
        output , (hidden,cell) = self.lstm(x)
        hidden = hidden[-1]
        x = self.fc(hidden)
        return x

if __name__ == "__main__":
    
    model = SentimentLSTM(
        vocab_size = 20002,
        embedding_dim = 128,
        hidden_size = 128,
        num_classes = 2
    )
    x = torch.tensor([[11,66,450,12,19]])

    output = model(x)

    print("Input Shape: " , x.shape)
    print("Output Shape: " , output.shape)