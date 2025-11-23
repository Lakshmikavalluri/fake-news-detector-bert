import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import numpy as np
from tqdm import tqdm

# -----------------------------
# 1. Load the Dataset
# -----------------------------
df1 = pd.read_csv("data/news.csv")
df2 = pd.read_csv("data/news_dataset.csv")

df = pd.concat([df1, df2], ignore_index=True)

# Convert labels to numeric
df["label"] = df["label"].str.upper()
df["label"] = df["label"].map({"REAL": 0, "FAKE": 1})

# Drop missing rows
df = df.dropna(subset=["text", "label"])

print("Dataset loaded:")
print(df["label"].value_counts())

# -----------------------------
# 2. Train-Test Split
# -----------------------------
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42,
)

# -----------------------------
# 3. Load Tokenizer
# -----------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# -----------------------------
# 4. Create PyTorch Dataset
# -----------------------------
class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=256,
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = NewsDataset(train_texts, train_labels)
test_dataset = NewsDataset(test_texts, test_labels)

# -----------------------------
# 5. DataLoader
# -----------------------------
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# -----------------------------
# 6. Load BERT Model
# -----------------------------
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

# -----------------------------
# 7. Training Loop
# -----------------------------
EPOCHS = 2
model.train()

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    for batch in tqdm(train_loader):
        optimizer.zero_grad()

        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

print("\nTraining complete!")

# -----------------------------
# 8. Evaluation
# -----------------------------
model.eval()
preds = []
true_labels = []

with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        logits = outputs.logits
        pred_labels = torch.argmax(logits, dim=1)

        preds.extend(pred_labels.cpu().numpy())
        true_labels.extend(batch["labels"].cpu().numpy())

print("\nEvaluation Report:")
print(classification_report(true_labels, preds))

# -----------------------------
# 9. Save the Trained Model
# -----------------------------
os.makedirs("model/bert", exist_ok=True)
model.save_pretrained("model/bert")
tokenizer.save_pretrained("model/bert")

print("\nBERT model saved to: model/bert")
