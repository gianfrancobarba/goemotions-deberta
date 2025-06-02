# File: features.py

"""
features.py

Estrazione di feature interpretabili da una lista di testi:
- Bag-of-Words locale
- Conteggio di parole emozionali (NRC) suddivise per categoria
- Negazioni
- Statistiche strutturali (lunghezza, stopword ratio, punteggiatura)
"""

import re
import os
from typing import List, Dict, Set, Optional, Union

import pandas as pd
from nltk.corpus import stopwords


stop_words = set(stopwords.words("english"))
NEGATION_WORDS = {"not", "never", "no", "none", "nobody", "nothing", "neither", "nor"}


def load_nrc_lexicon(
    filepath: str,
    selected_emotions: Optional[List[str]] = None
) -> Dict[str, Set[str]]:
    """
    Carica il dizionario NRC da file.
    Ritorna: mapping emotion -> set(parole).

    Args:
      filepath: percorso al file NRC.
      selected_emotions: lista di emozioni da includere; se None, include tutte.
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
        - emo_<emotion>_count: numero di parole del lessico NRC per ciascuna emozione
        - n_emotion_words: somma di tutti i conteggi emozionali
        - has_negation: 0/1 se c'è almeno una negazione
        - length: numero di token puliti
        - n_exclamations: numero di '!'
        - n_questions: numero di '?'
        - stopword_ratio: rapporto tra stopword e lunghezza del testo
    """
    # 1) Caricamento lessico NRC (tutte le emozioni)
    lex_path = os.path.join(
        os.path.dirname(__file__),
        "lexicon",
        "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
    )
    nrc = load_nrc_lexicon(lex_path, selected_emotions=None)

    # 2) Costruzione vocabolario locale e tokenizzazione
    vocab: Set[str] = set()
    tokenized: List[List[str]] = []
    for txt in texts:
        toks = re.findall(r"\b\w+\b", txt.lower())
        # tolgo punteggiatura e stopword (ma mantengo negazioni)
        clean = [t for t in toks if t.isalpha() and (t not in stop_words or t in NEGATION_WORDS)]
        vocab.update(clean)
        tokenized.append(clean)

    vocab_list = sorted(vocab)
    rows: List[Dict[str, Union[int, float]]] = []

    for txt, tokens in zip(texts, tokenized):
        # 2.1) BOW (Bag‐of‐Words locale)
        bow_feats = {f"bow_{w}": int(w in tokens) for w in vocab_list}

        # 2.2) Conteggio parole emozionali per ciascuna emozione NRC
        #      e somma totale di parole emozionali (n_emotion_words)
        emo_counts: Dict[str, int] = {}
        total_emo = 0
        for emo_label, lex_words in nrc.items():
            count_this = sum(1 for w in tokens if w in lex_words)
            emo_counts[f"emo_{emo_label}_count"] = count_this
            total_emo += count_this

        # 2.3) Negazioni
        has_neg = int(any(w in NEGATION_WORDS for w in tokens))

        # 2.4) Statistiche strutturali
        length = len(tokens)
        n_excl = txt.count("!")
        n_q = txt.count("?")
        # stopword_ratio = stopword / lunghezza (case-insensitive)
        stop_ratio = (
            sum(1 for w in re.findall(r"\b\w+\b", txt.lower()) if w in stop_words)
            / max(length, 1)
        )

        row: Dict[str, Union[int, float]] = {
            **bow_feats,
            # Feature emozionali
            "n_emotion_words": total_emo,
            **emo_counts,
            # Negazioni
            "has_negation": has_neg,
            # Strutturali
            "length": length,
            "n_exclamations": n_excl,
            "n_questions": n_q,
            "stopword_ratio": round(stop_ratio, 3),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if as_dict:
        return rows
    return df
