# ml/sentiment.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

_tokenizer = None
_model = None

LABELS = ["negative", "neutral", "positive"]


def _load_model():
    global _tokenizer, _model

    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    return _tokenizer, _model


def analyze_sentiment(text: str):
    if not text or not text.strip():
        return {
            "label": "neutral",
            "score": 0,
            "raw": {}
        }

    tokenizer, model = _load_model()

    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0].tolist()
    label_id = int(torch.argmax(outputs.logits, dim=1))

    return {
        "label": LABELS[label_id],
        "score": float(probs[label_id]),
        "raw": {
            "negative": float(probs[0]),
            "neutral": float(probs[1]),
            "positive": float(probs[2])
        }
    }
