import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

# --- BertSentimentClassifier Definition ---

class BertSentimentClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.bert = AutoModel.from_pretrained("bert-base-uncased")

        # Freeze the bottom 8 of BERT's 12 encoder layers (plus embeddings).
        # Lower layers mostly learn generic language features that transfer
        # fine without fine-tuning; only the top layers need to adapt to
        # sentiment specifically. This cuts backward-pass cost substantially
        # since gradients don't need to be computed/stored for frozen params.
        # If you want max accuracy over max speed, comment this block out.
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[:8]:
            for param in layer.parameters():
                param.requires_grad = False

        self.classifier = nn.Linear(
            self.bert.config.hidden_size,
            num_classes
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls_output = output.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits

# --- Data Loading ---

dataset = load_dataset("stanfordnlp/imdb")

train_data = dataset["train"]
test_data = dataset["test"]

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

MAX_LENGTH = 256  # 300 -> 256 covers the vast majority of IMDB reviews and cuts compute ~15%

# ---------------------------------------------------------------
# KEY FIX #1: Tokenize once, up front, in batched fashion via
# dataset.map(), instead of re-running the (slow) Python tokenizer
# on a single string inside __getitem__ on every single access.
# With your original code, every epoch re-tokenizes all 25k
# reviews one-at-a-time, on CPU, every time DataLoader pulls a
# sample. That's pure wasted work repeated 3 times (once per
# epoch) instead of once total.
# ---------------------------------------------------------------

def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        # No padding here - we pad per-batch instead (see KEY FIX #2)
    )

train_data = train_data.map(tokenize_fn, batched=True, remove_columns=["text"])
test_data = test_data.map(tokenize_fn, batched=True, remove_columns=["text"])

# NOTE: We deliberately do NOT call dataset.set_format(type="torch") here.
# On Colab, that triggers the datasets library's torch formatter to probe
# for torchvision (to handle image/video columns), and Colab's current
# torchvision build is mismatched with torch, causing:
#   ImportError: cannot import name 'VideoReader' from 'torchvision.io'
# We don't need set_format at all - we convert to tensors ourselves in
# collate_fn below, so the dataset can stay in plain Python/list form.

# ---------------------------------------------------------------
# KEY FIX #2: Dynamic padding per batch instead of padding every
# review to a fixed MAX_LENGTH. Most IMDB reviews are much shorter
# than 256/300 tokens, so padding every sample to the max wastes
# a large fraction of every matrix multiply in the model on tokens
# that carry no information. Padding only to the longest sequence
# in each batch (via a collate function) means most batches run
# on sequences far shorter than 256, which is a large speedup.
# ---------------------------------------------------------------

def collate_fn(batch):
    # batch items are plain dicts of Python lists/ints here (dataset is NOT
    # set_format("torch")), so we build the label tensor ourselves too.
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    token_type_ids = [item["token_type_ids"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    padded = tokenizer.pad(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
        padding=True,
        return_tensors="pt",
    )

    return {
        "input_ids": padded["input_ids"],
        "attention_mask": padded["attention_mask"],
        "token_type_ids": padded["token_type_ids"],
        "label": labels,
    }

BATCH_SIZE = 32  # bumped from 16 - fp16 + dynamic padding leaves headroom on a T4 at max_length=256

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=2,      # overlap batch prep with GPU compute
    pin_memory=True,    # speeds up host->device transfer
)

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE * 2,  # no gradients needed at eval, so we can go bigger
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True,
)

# --- Hyperparameters ---

LEARNING_RATE = 2e-5
EPOCHS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.backends.cudnn.benchmark = True  # lets cuDNN pick faster kernels once input shapes stabilize

model = BertSentimentClassifier(num_classes=2)
model = model.to(device)

# torch.compile fuses ops and cuts Python/CUDA-launch overhead - typically
# a free 20-40% speedup on PyTorch 2.0+, no accuracy impact. If your PyTorch
# version is older than 2.0 or this errors on your setup, just delete this line.
model = torch.compile(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)

# ---------------------------------------------------------------
# KEY FIX #3: Mixed precision (fp16) via autocast + GradScaler.
# This is the single biggest speedup available on a T4 - T4s have
# dedicated Tensor Cores that are roughly 2-3x faster at fp16
# matmuls than fp32, and it also roughly halves memory usage,
# letting you use a larger batch size if you want to push it later.
# ---------------------------------------------------------------

scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            token_type_ids = batch["token_type_ids"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        token_type_ids = batch["token_type_ids"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # slightly faster than zeroing tensors

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{EPOCHS}] Loss: {average_loss:.4f}")

test_accuracy = evaluate(test_loader)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
