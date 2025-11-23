import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load saved model and tokenizer
model_path = "model/bert"

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def predict_bert(text: str):
    """
    Returns:
    - label: 0 = REAL, 1 = FAKE
    - fake_probability: float (0–1)
    """

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)

    fake_prob = float(probs[0][1])
    label = 1 if fake_prob > 0.5 else 0

    return label, fake_prob
