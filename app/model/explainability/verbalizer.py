import os
from typing import Dict, Any, List, Set

from model.explainability.contrastive import get_contrastive_deltas

# =======================
# EMOTION VOCAB
# =======================
LEXICON_PATH = os.path.join(
    os.path.dirname(__file__),
    "lexicon",
    "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
)

def load_nrc_vocab(path: str) -> Set[str]:
    vocab: Set[str] = set()
    if not os.path.exists(path):
        return vocab
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            word, emotion, score_str = parts
            try:
                score = int(score_str)
            except ValueError:
                continue
            if score > 0:
                vocab.add(word.lower())
    return vocab

# Carica NRC e unisce parole aggiuntive hardcoded
EMOTION_VOCAB: Set[str] = load_nrc_vocab(LEXICON_PATH)
ADDITIONAL_EMOTION_WORDS = {
    "proud", "ecstatic", "devastated", "furious", "overjoyed", "terrified",
    "enthusiastic", "hopeful", "anxious", "grateful", "ashamed"
}
EMOTION_VOCAB.update(ADDITIONAL_EMOTION_WORDS)

# =======================
# UTILITY
# =======================
def count_words(text: str) -> int:
    return len(text.strip().split())

def count_exclamations(text: str) -> int:
    return text.count("!")

def count_questions(text: str) -> int:
    return text.count("?")

def extract_emotion_words(text: str, emotion_vocab: Set[str]) -> List[str]:
    tokens = [w.lower().strip(".,!?:;\"'()[]") for w in text.split()]
    return [tok for tok in tokens if tok in emotion_vocab]

def tokenize(text: str) -> List[str]:
    return [w.lower().strip(".,!?:;\"'()[]") for w in text.split()]

def parse_rules(rules_str: str) -> str:
    lines = rules_str.splitlines()
    parsed_lines: List[str] = []
    for line in lines:
        depth = line.count("|")
        content = line.split("---", 1)[1].strip() if "---" in line else line.strip()
        content = content.replace("  ", " ").strip().replace("bow_", "")
        indent = "  " * max(0, depth - 1)
        parsed_lines.append(f"{indent}- {content}")
    return "\n".join(parsed_lines)

# =======================
# VERBALIZER
# =======================
def verbalize_explanation(result: Dict[str, Any]) -> Dict[str, Any]:
    explanations: Dict[str, Any] = {}
    predictions    = result.get("predictions", {})
    features_list  = result.get("features", [])
    surrogates     = result.get("surrogates", {})
    original_text  = result.get("original_text", "")
    base_feats     = features_list[0] if features_list else {}

    word_count = count_words(original_text)
    n_excl     = count_exclamations(original_text)
    n_qst      = count_questions(original_text)
    stop_ratio = base_feats.get("stopword_ratio", None)
    stop_ratio_rounded = round(stop_ratio, 2) if stop_ratio is not None else None
    has_neg = bool(base_feats.get("has_negation", 0))

    for emo, data in surrogates.items():
        prob        = predictions.get(emo, 0.0)
        metrics     = data.get("metrics", {})
        importances = data.get("feature_importances", {})
        raw_rules   = data.get("rules", "").strip()

        tokens = tokenize(original_text)
        token_set = set(tokens)
        emotion_words = [tok for tok in tokens if tok in EMOTION_VOCAB]

        surrogate_keywords: Set[str] = set()
        for feat, imp in importances.get("bow", {}).items():
            if isinstance(imp, (float, int)) and imp > 0:
                word = feat.replace("bow_", "")
                if word in token_set:
                    surrogate_keywords.add(word)

        combined_emotion_words = sorted(
            set(emotion_words),
            key=lambda w: tokens.index(w) if w in tokens else 999
        )

        active_bow = [
            (feat.replace("bow_", ""), imp)
            for feat, imp in importances.get("bow", {}).items()
            if isinstance(imp, (float, int)) and imp > 0
        ]
        active_bow.sort(key=lambda x: x[1], reverse=True)

        top_keywords = [word for word, _ in active_bow if word in token_set][:3]
        if not top_keywords and active_bow:
            top_keywords = [active_bow[0][0]]

        frasi: List[str] = []
        emocapital = emo.upper()
        frasi.append(f"Il modello ha classificato il testo come “{emocapital}” (probabilità: {prob:.2f}).")
        if top_keywords:
            frasi.append(f"La parola chiave più rilevante è “{top_keywords[0]}”.")
        else:
            frasi.append("Non ci sono parole chiave particolarmente influenti.")
        frasi.append(f"Il testo è composto da {word_count} {'parola' if word_count == 1 else 'parole'}.")
        if has_neg:
            frasi.append("Nel testo è presente almeno una negazione (‘non’).")
        if n_excl > 0:
            frasi.append(f"Sono presenti {n_excl} {'punto esclamativo' if n_excl == 1 else 'punti esclamativi'}.")
        if n_qst > 0:
            frasi.append(f"Sono presenti {n_qst} {'punto interrogativo' if n_qst == 1 else 'punti interrogativi'}.")
        if stop_ratio_rounded is not None:
            frasi.append(f"Il rapporto stop-word è {stop_ratio_rounded:.2f} (rapporto tra parole neutre e parole emotive).")
        if combined_emotion_words:
            frasi.append(f"Le parole emotive rilevate nel testo sono: {', '.join(combined_emotion_words)}.")
        else:
            frasi.append("Nessuna parola emotiva rilevata.")
        if top_keywords:
            frasi.append(f"In sintesi, la presenza di “{top_keywords[0]}” e la lunghezza del testo hanno portato il modello a predire “{emocapital}”.")
        else:
            frasi.append(f"In sintesi, segnali come lunghezza, stop-word e punteggiatura hanno portato il modello a predire “{emocapital}”.")

        contrastive_deltas = get_contrastive_deltas(
            original_text,
            emotion=emo,
            top_words=top_keywords
        )
        if contrastive_deltas:
            frasi.append("Analisi contrastiva:")
            for word, delta in contrastive_deltas.items():
                original = predictions.get(emo, 0.0)
                perturbed = original - delta
                frasi.append(f"Rimuovendo “{word}”, la probabilità di {emocapital} scende da {original:.2f} a {perturbed:.2f}.")

        simple_explanation = "\n".join(frasi)

        all_imps = sorted(
            [
                (feat.replace("bow_", ""), round(imp, 4))
                for feat, imp in importances.get("bow", {}).items()
                if isinstance(imp, (float, int)) and imp > 0
            ],
            key=lambda x: x[1], reverse=True
        )

        parsed = parse_rules(raw_rules) if raw_rules else ""

        explanations[emo] = {
            "simple_explanation": simple_explanation,
            "emotion_words": combined_emotion_words,
            "emotion": emo,
            "probability": round(prob, 2),
            "top_keywords": top_keywords,
            "word_count": word_count,
            "n_emotion_words": base_feats.get("n_emotion_words", None),
            "has_negation": has_neg,
            "stopword_ratio": stop_ratio_rounded,
            "exclamation_count": n_excl,
            "question_count": n_qst,
            "metrics": {
                "fidelity": round(metrics.get("fidelity", 0.0), 2),
                "sparsity": round(metrics.get("sparsity", 0.0), 2),
                "stability": round(metrics.get("stability", 0.0), 2),
            },
            "all_importances": all_imps,
            "raw_rules": raw_rules,
            "parsed_rules": parsed,
        }

    return explanations
