# predictor.py

from typing import List, Dict
import torch
from model.inference.predict import tokenizer, model, device, GOEMOTIONS_LABELS
from config_surrogate import MAX_LEN, THRESHOLD

def predict_all(texts: List[str]) -> List[Dict[str, float]]:
    """
    Prende una lista di testi e restituisce, per ciascuno, un dizionario
    con le probabilità (0.0–1.0) di tutte le emozioni.
    """
    results = []
    model.eval()

    for text in texts:
        encoding = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)["logits"]
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        prediction = {
            label: round(float(prob), 3)
            for label, prob in zip(GOEMOTIONS_LABELS, probs)
        }
        results.append(prediction)

    return results


# ====================
# TEST AUTOMATICO
# ====================
if __name__ == "__main__":
    test_inputs = [
        "I'm very happy with how things turned out!",
        "I feel empty and lost.",
        "Why did this happen to me?"
    ]

    print("\n=== TEST predict_all() ===\n")
    outputs = predict_all(test_inputs)
    for i, output in enumerate(outputs):
        print(f"Text {i}:")
        for emotion, score in output.items():
            if score > THRESHOLD:
                print(f"  {emotion}: {score}")
        print()
