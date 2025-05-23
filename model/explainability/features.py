# features.py

import re
import os
import pandas as pd
from typing import List, Optional, Dict, Set
from collections import Counter

import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

NEGATION_WORDS = {"not", "never", "no", "none", "nobody", "nothing", "neither", "nor"}

def load_nrc_lexicon(filepath: str, selected_emotions: List[str] = None) -> Dict[str, Set[str]]:
    """
    Carica il dizionario NRC.
    Ritorna: {emotion: set(parole)}
    """
    lexicon = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue  # ignora righe malformattate
            word, emotion, assoc = parts
            if int(assoc) == 1:
                if selected_emotions is None or emotion in selected_emotions:
                    lexicon.setdefault(emotion, set()).add(word)
    return lexicon


def extract_features(texts: List[str], output_csv: Optional[str] = None) -> pd.DataFrame:
    """
    Estrae caratteristiche interpretabili da una lista di frasi.
    Opzionalmente salva un .csv in 'tests/'.
    """
    # === CARICA NRC ===
    LEXICON_PATH = os.path.join(os.path.dirname(__file__), "lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")
    selected_emotions = ["anger", "fear", "joy", "sadness", "disgust", "surprise", "trust", "anticipation"]
    nrc_lexicon = load_nrc_lexicon(LEXICON_PATH, selected_emotions)

    vocab = set()
    tokenized_texts = []

    for text in texts:
        tokens = re.findall(r"\b\w+\b", text.lower())
        tokens_clean = [t for t in tokens if t.isalpha() and (t not in stop_words or t in NEGATION_WORDS)]
        vocab.update(tokens_clean)
        tokenized_texts.append(tokens_clean)

    vocab = sorted(vocab)
    data = []

    for i, tokens in enumerate(tokenized_texts):
        text = texts[i]
        bow = {f"bow_{word}": int(word in tokens) for word in vocab}

        # === SEMANTICHE: parole emozionali (da NRC)
        emotion_count = 0
        for word in tokens:
            for emotion_words in nrc_lexicon.values():
                if word in emotion_words:
                    emotion_count += 1
                    break

        negation_present = any(w in NEGATION_WORDS for w in tokens)

        # === STRUTTURALI
        length = len(tokens)
        exclamations = text.count("!")
        questions = text.count("?")
        stopword_ratio = len([w for w in re.findall(r"\b\w+\b", text.lower()) if w in stop_words]) / max(1, length)

        row = {
            **bow,
            "n_emotion_words": emotion_count,
            "has_negation": int(negation_present),
            "length": length,
            "n_exclamations": exclamations,
            "n_questions": questions,
            "stopword_ratio": round(stopword_ratio, 3)
        }

        data.append(row)

    df = pd.DataFrame(data)

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)

    return df


# =====================
# TEST AUTOMATICO
# =====================
if __name__ == "__main__":
    test_texts = [
        "I am so happy and excited to be here!",
        "I am not sure about this...",
        "No one ever listens to me.",
        "I'm proud of myself!",
        "Why did I even try?"
    ]

    df = extract_features(test_texts, output_csv="tests/features_test.csv")
    print("\n=== TEST extract_features() ===\n")
    print(df.head())
