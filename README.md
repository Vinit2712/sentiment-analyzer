# Sentiment Analyzer

A sentiment analysis project exploring the evolution of Natural Language Processing (NLP), from traditional statistical methods to modern Transformer-based models.

The project compares three approaches on the IMDb movie review dataset:

1. TF-IDF + Logistic Regression
2. Word Embeddings + LSTM
3. Pretrained BERT + Fine-Tuning

The goal was not only to build a sentiment classifier, but also to understand how NLP representations and architectures evolved from simple word statistics to contextual representations using self-attention.

---

## Results

| Model | Approach | Test Accuracy |
|---|---|---:|
| TF-IDF + Logistic Regression | Traditional NLP | **88.29%** |
| Embedding + LSTM | Neural NLP | **84.19%** |
| BERT + Fine-Tuning | Modern NLP | **92.03%** |

BERT achieved the highest accuracy in this experiment.

> **Note:** These models use different architectures, capacities, and training configurations, so these results reflect the performance of these particular implementations rather than proving that one architecture is universally superior to another.

---

## Dataset

The project uses the **IMDb Movie Reviews** dataset (`stanfordnlp/imdb`).

- 25,000 training reviews
- 25,000 test reviews
- Binary sentiment classification
  - `0` → Negative
  - `1` → Positive

---

## 1. TF-IDF + Logistic Regression

The first approach uses traditional NLP techniques.

**Pipeline**

```
Raw Review → Tokenization → TF-IDF → Feature Vector → Logistic Regression → Positive / Negative
```

**TF-IDF** assigns importance to words based on:

- How frequently a word appears in a document
- How common or uncommon the word is across the whole dataset

Common words such as `the`, `and`, and `is` receive lower importance, while words that better distinguish one document from another receive higher scores.

**Result: 88.29% test accuracy**

---

## 2. Word Embeddings + LSTM

The second approach moves from statistical NLP to neural NLP. Instead of representing words using TF-IDF scores, words are mapped to learned dense vectors.

**Pipeline**

```
Token → Token ID → Embedding → LSTM → Hidden State → Linear Layer → Positive / Negative
```

**Configuration**

- Vocabulary size: 20,002
- Embedding dimension: 128
- Hidden size: 128
- Maximum sequence length: 300
- 2 output classes

**What the LSTM adds over TF-IDF**

Unlike TF-IDF, the LSTM processes tokens sequentially and maintains a hidden state, allowing it to use information from earlier words when processing later ones:

```
The movie was not good
                  ↑
          previous context
```

The LSTM was trained from scratch on the IMDb dataset.

**Result: 84.19% test accuracy**

---

## 3. BERT (Fine-Tuned)

The final approach uses modern Transformer-based NLP, built on `bert-base-uncased`.

Instead of manually building a vocabulary and padding system, BERT's tokenizer handles tokenization, subword splitting, special tokens, token IDs, padding, attention masks, and token-type IDs directly.

**Pipeline**

```
Raw Review → BERT Tokenizer → Input IDs / Attention Mask / Token Type IDs
           → Pretrained BERT → [CLS] Representation → Linear Layer → Positive / Negative
```

### How BERT works

BERT is based on the Transformer architecture. The input representation combines:

```
Token Embedding + Position Embedding + Token-Type Embedding
```

This passes through multiple Transformer encoder layers, each containing:

```
Multi-Head Self-Attention → Add & Norm → Feed-Forward Network → Add & Norm
```

**Self-attention** lets each token weigh every other token in the sequence when building its representation, rather than treating words independently:

```
The movie was not good
              ↑
              |
       not ← good
```

**Multi-head attention** runs several attention mechanisms in parallel, so different heads can learn different relationships between tokens.

**Query, Key, Value**

Self-attention builds three representations per token — Query (Q), Key (K), and Value (V) — and computes attention scores as:

```
softmax(QKᵀ / √dk)
```

These scores determine how much information each token pulls from the others.

### Pretraining

BERT is pretrained before being fine-tuned on sentiment. One of its original objectives is **Masked Language Modeling** — predicting a hidden word from context, e.g.:

```
The movie was really [MASK].   →   good
```

This lets BERT learn general-purpose language representations from large text corpora without needing labeled data for every example.

### Fine-tuning

A `[CLS]` token is placed at the start of the input:

```
[CLS] The movie was really good [SEP]
```

After passing through the Transformer layers, the final `[CLS]` representation summarizes the whole sequence and feeds into the classification head:

```
768-dim [CLS] vector → Linear Layer → 2 outputs → Negative / Positive
```

The model was fine-tuned end-to-end on the IMDb training set.

**Result: 92.03% test accuracy**

---

## Comparing the Three Approaches

The project traces a progression in how text is represented:

| Stage | Representation | Example |
|---|---|---|
| Traditional NLP | Statistical word importance | `"good"` → numerical weight |
| Neural NLP | Learned dense vector | `"good"` → embedding |
| Modern NLP | Contextual representation | `"good"` → meaning shaped by surrounding words |

With BERT, the same word can produce different representations depending on how it's used in context — something neither TF-IDF nor a standard embedding layer can do.

---

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Scikit-learn
- NumPy
- BERT · LSTM · TF-IDF · Logistic Regression

---

## Project Structure

```
sentiment-analyzer/
│
├── src/
│   ├── baseline.py
│   ├── bert_train.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── model.py
│   └── train.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Vinit2712/sentiment-analyzer.git
cd sentiment-analyzer
```

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS / Linux
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## What I Learned

This project walks through the progression of NLP from traditional statistical methods to modern Transformer-based models, covering:

- Tokenization, vocabulary creation, and token IDs
- TF-IDF and logistic regression for text classification
- Word embeddings, padding, and truncation
- LSTM hidden states and backpropagation through embeddings
- Self-attention, Query/Key/Value, and multi-head attention
- Positional and token-type embeddings
- Transformer encoder blocks
- BERT pretraining (Masked Language Modeling) and fine-tuning
- Contextual word representations

---

## Conclusion

This project started with a simple statistical representation of text using TF-IDF and progressed through recurrent neural networks to Transformer-based NLP:

```
TF-IDF → LSTM → Transformer → BERT
```

The final fine-tuned BERT model achieved **92.03% test accuracy** on the IMDb sentiment classification task.
