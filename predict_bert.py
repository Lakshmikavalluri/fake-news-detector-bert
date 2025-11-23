import torch
from transformers import BertTokenizer, BertForSequenceClassification


# --------------------------------------------------
# 🔥 Load BERT model from Hugging Face (Online)
# --------------------------------------------------
MODEL_NAME = "lakshmikavalluri123/fake-news-bert-lakshmika"

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)

model.eval()   # Set model to evaluation mode


# --------------------------------------------------
# 🔍 Prediction Function
# --------------------------------------------------
def predict_bert(text):

    if not text or text.strip() == "":
        return 1, 0.99  # Treat empty text as high fake probability

    # Tokenize
    encoded_input = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**encoded_input)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)

    # Probability of fake news (label 1)
    fake_prob = float(probs[0][1])

    # Prediction label (0=real, 1=fake)
    label = int(torch.argmax(probs, dim=1).item())

    return label, fake_prob
