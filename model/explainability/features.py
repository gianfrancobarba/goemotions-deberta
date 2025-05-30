# features.py

"""
features.py

Estrazione di feature interpretabili da una lista di testi:
- Bag-of-Words locale
- Conteggio di parole emozionali (NRC)
- Negazioni
- Statistiche strutturali (lunghezza, stopword ratio, punteggiatura)
"""

import re
import os
from typing import List, Dict, Set, Optional, Union

import pandas as pd
from nltk.corpus import stopwords

# Non scarica nulla in runtime; presuppone che i dati nltk siano già installati
stop_words = set(stopwords.words("english"))
NEGATION_WORDS = {"not", "never", "no", "none", "nobody", "nothing", "neither", "nor"}


def load_nrc_lexicon(
    filepath: str,
    selected_emotions: Optional[List[str]] = None
) -> Dict[str, Set[str]]:
    """
    Carica il dizionario NRC da file.
    Ritorna: mapping emotion -> set(parole).
    """
    lexicon: Dict[str, Set[str]] = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            word, emotion, assoc = parts
            if assoc == "1" and (selected_emotions is None or emotion in selected_emotions):
                lexicon.setdefault(emotion, set()).add(word)
    return lexicon


def extract_features(
    texts: List[str],
    as_dict: bool = False
) -> Union[pd.DataFrame, List[Dict[str, Union[int, float]]]]:
    """
    Estrae feature interpretabili da una lista di frasi.

    Args:
      texts: lista di frasi in input.
      as_dict: se True restituisce List[Dict], altrimenti pandas.DataFrame.

    Returns:
      DataFrame o lista di dict con:
        - bow_<term>: 0/1 per ogni parola del vocabolario locale
        - n_emotion_words, has_negation, length, n_exclamations,
          n_questions, stopword_ratio
    """
    # Caricamento lessico NRC
    lex_path = os.path.join(
        os.path.dirname(__file__),
        "lexicon",
        "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
    )
    selected = ["anger", "fear", "joy", "sadness", "disgust", "surprise",
                "trust", "anticipation"]
    nrc = load_nrc_lexicon(lex_path, selected)

    # Costruzione vocabolario locale e tokenizzazione
    vocab: Set[str] = set()
    tokenized: List[List[str]] = []
    for txt in texts:
        toks = re.findall(r"\b\w+\b", txt.lower())
        clean = [t for t in toks if t.isalpha() and (t not in stop_words or t in NEGATION_WORDS)]
        vocab.update(clean)
        tokenized.append(clean)

    vocab_list = sorted(vocab)
    rows: List[Dict[str, Union[int, float]]] = []

    for txt, tokens in zip(texts, tokenized):
        # BOW
        bow_feats = {f"bow_{w}": int(w in tokens) for w in vocab_list}

        # Semantiche (NRC)
        emo_count = 0
        for w in tokens:
            if any(w in words for words in nrc.values()):
                emo_count += 1

        # Negazione
        has_neg = int(any(w in NEGATION_WORDS for w in tokens))

        # Strutturali
        length = len(tokens)
        n_excl = txt.count("!")
        n_q = txt.count("?")
        stop_ratio = sum(1 for w in re.findall(r"\b\w+\b", txt.lower())
                         if w in stop_words) / max(length, 1)

        row = {
            **bow_feats,
            "n_emotion_words": emo_count,
            "has_negation": has_neg,
            "length": length,
            "n_exclamations": n_excl,
            "n_questions": n_q,
            "stopword_ratio": round(stop_ratio, 3)
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if as_dict:
        return rows
    return df
