# perturber.py

"""
    Obiettivo: Generare un insieme di testi perturbati semanticamente simili a un commento dato,
    per costruire un neighborhood locale attorno all’istanza da spiegare.
    Questi testi saranno utilizzati per addestrare il modello surrogato interpretabile.
"""

import random
import re
from typing import List

import nltk
from nltk.corpus import wordnet, stopwords
from nltk import pos_tag

# Scarica solo i pacchetti sicuri e necessari
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))


def generate_perturbations(text: str, n: int = 100, strategies: List[str] = None) -> List[str]:
    """
    Genera una lista di frasi perturbate semanticamente simili.
    Include sempre il testo originale come prima riga.
    """
    if strategies is None:
        strategies = ["dropout", "masking", "shuffle", "synonym"]

    perturbations = [text]
    while len(perturbations) < n:
        perturbed = apply_random_strategy(text, strategies)
        if perturbed not in perturbations:
            perturbations.append(perturbed)
    return perturbations


def apply_random_strategy(text: str, strategies: List[str]) -> str:
    """
    Applica una strategia casuale scelta tra quelle fornite.
    """
    strategy = random.choice(strategies)
    if strategy == "dropout":
        return word_dropout(text, p=random.uniform(0.3, 0.6))
    elif strategy == "masking":
        return word_masking(text, p=random.uniform(0.2, 0.4))
    elif strategy == "shuffle":
        return slight_shuffle(text)
    elif strategy == "punctuation_removal":
        return re.sub(r"[^\w\s]", "", text)
    elif strategy == "synonym":
        return synonym_replacement(text, p=0.5)
    else:
        return text


def word_dropout(text: str, p: float) -> str:
    words = text.split()
    return " ".join([w for w in words if random.random() > p])


def word_masking(text: str, p: float) -> str:
    words = text.split()
    masked = [("[MASK]" if random.random() < p else w) for w in words]
    return " ".join(masked)


def slight_shuffle(text: str) -> str:
    words = text.split()
    if len(words) <= 2:
        return text
    i = random.randint(0, len(words) - 2)
    words[i], words[i + 1] = words[i + 1], words[i]
    return " ".join(words)


def get_wordnet_pos(treebank_tag):
    """
    Mappa i tag POS da Treebank a WordNet.
    """
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return None


def synonym_replacement(text: str, p: float = 0.3) -> str:
    """
    Sostituisce p% delle parole (non-stopword, alfabetiche) con sinonimi reali.
    Compatibile con NLTK >= 3.8.2 (senza punkt).
    """
    # Tokenizzazione compatibile: solo parole alfabetiche
    tokens = re.findall(r"\b\w+\b", text)
    tagged = pos_tag(tokens, lang="eng")

    new_tokens = []
    for i, (word, tag) in enumerate(tagged):
        wn_tag = get_wordnet_pos(tag)
        if wn_tag is None or word.lower() in stop_words or not word.isalpha():
            new_tokens.append(word)
            continue

        if random.random() < p:
            synonyms = wordnet.synsets(word, pos=wn_tag)
            if not synonyms:
                new_tokens.append(word)
                continue

            lemmas = set()
            for syn in synonyms:
                for l in syn.lemmas():
                    lemma = l.name().replace("_", " ")
                    if lemma.lower() != word.lower():
                        lemmas.add(lemma)

            if lemmas:
                replacement = random.choice(list(lemmas))
                new_tokens.append(replacement)
            else:
                new_tokens.append(word)
        else:
            new_tokens.append(word)

    return " ".join(new_tokens)


# =======================
# TEST DI FUNZIONAMENTO
# =======================
if __name__ == "__main__":
    text = "I finally did it, I'm so proud of myself!"
    print("\n=== TEST generate_perturbations() ===\n")
    perturbed = generate_perturbations(text, n=10, strategies=["dropout", "masking", "shuffle", "synonym"])
    for i, t in enumerate(perturbed):
        print(f"{i}: {t}")
