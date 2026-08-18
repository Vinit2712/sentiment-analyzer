import re
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset



dataset = load_dataset("stanfordnlp/imdb")

train_data = dataset["train"]
test_data = dataset["test"]



def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())



counter = Counter()

for review in train_data["text"]:
    tokens = tokenize(review)
    counter.update(tokens)


MAX_VOCAB_SIZE = 20_000

vocab = {
    "<PAD>": 0,
    "<UNK>": 1
}

for word, count in counter.most_common(MAX_VOCAB_SIZE):
    vocab[word] = len(vocab)



def encode(text):
    tokens = tokenize(text)

    ids = []

    for token in tokens:
        if token in vocab:
            ids.append(vocab[token])
        else:
            ids.append(vocab["<UNK>"])

    return ids



MAX_LENGTH = 150


def pad_or_truncate(ids):

    if len(ids) > MAX_LENGTH:
        return ids[:MAX_LENGTH]

    if len(ids) < MAX_LENGTH:
        padding = [vocab["<PAD>"]] * (MAX_LENGTH - len(ids))
        return ids + padding

    return ids



class IMDBDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        review = self.data[index]["text"]
        label = self.data[index]["label"]

        ids = encode(review)
        ids = pad_or_truncate(ids)

        ids = torch.tensor(ids, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)

        return ids, label



train_dataset = IMDBDataset(train_data)
test_dataset = IMDBDataset(test_data)



train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)



if __name__ == "__main__":

    reviews, labels = next(iter(train_loader))

    print("Batch input shape:", reviews.shape)
    print("Batch labels shape:", labels.shape)