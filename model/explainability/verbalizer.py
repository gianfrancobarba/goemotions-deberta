# File: model/explainability/verbalizer.py

import os
from typing import Dict, Any, List, Set

# ————————————————————————————————————————————
# 1) TENTATIVO DI CARICARE AUTOMATICAMENTE IL LESSICO NRC
#
# Se hai già eseguito download_nrc_lexicon.py e il file si trova in
#     model/explainability/lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt
# allora lo carichiamo per costruire EMOTION_VOCAB dinamicamente.
LEXICON_PATH = os.path.join(
    os.path.dirname(__file__),
    "lexicon",
    "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
)

def load_nrc_vocab(path: str) -> Set[str]:
    """
    Legge il file NRC-Emotion-Lexicon (word-level) e restituisce
    l’insieme di tutte le parole che compaiono in almeno una riga con score 1.
    Il formato tipico (riga) è:
      abandoned	anger	1
      abandoned	negative	1
      abandoned	sadness	1
      abandonment	sadness	1
    Prende solo la prima colonna (la parola).
    """
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
            # Se score=1 (o >0) significa che “word” è associata a un’emozione
            if score > 0:
                vocab.add(word.lower())
    return vocab

# Costruisco EMOTION_VOCAB all’avvio, se il file esiste
EMOTION_VOCAB: Set[str] = load_nrc_vocab(LEXICON_PATH)

# Se per qualche motivo il lessico non si trova o è vuoto,
# posso aggiungere a mano qualche parola di fallback.
if not EMOTION_VOCAB:
    EMOTION_VOCAB = {
        "joy", "sad", "love", "anger", "fear", "surprise", "disgust",
        "pride", "grief", "relief", "gratitude", "hope", "anxiety",
        "amusement", "optimism", "admiration", "approval", "caring",
        "excitement", "contentment", "embarrassment", "nervousness",
        "disappointment", "remorse", "annoyance", "distressing",
        "lamentable", "proud",  # almeno questi…
    }
# ————————————————————————————————————————————

def count_words(text: str) -> int:
    """
    Contare quante parole (space-separated) ci sono nel testo.
    """
    return len(text.strip().split())

def count_exclamations(text: str) -> int:
    return text.count("!")

def count_questions(text: str) -> int:
    return text.count("?")

def extract_emotion_words(text: str, emotion_vocab: Set[str]) -> List[str]:
    """
    Ritorna la lista di token *presenti nel testo* che figurano in emotion_vocab.
    Rimuove punteggiatura di contorno e converte tutto in minuscolo.
    Mantiene l’ordine di comparsa nel testo.
    """
    tokens = [w.lower().strip(".,!?:;\"'()[]") for w in text.split()]
    return [tok for tok in tokens if tok in emotion_vocab]

def parse_rules(rules_str: str) -> str:
    """
    Converte l’output di export_text(clf) del DecisionTree in
    un elenco indentato di righe, togliendo il prefisso "bow_".
    """
    lines = rules_str.splitlines()
    parsed_lines: List[str] = []

    for line in lines:
        # Il numero di '|' prima di '---' corrisponde al livello di indentazione
        depth = line.count("|")
        if "---" in line:
            content = line.split("---", 1)[1].strip()
        else:
            content = line.strip()

        # Rimuovo doppi spazi e normalizzo
        content = content.replace("  ", " ").strip()
        # Tolgo prefisso "bow_" per rendere più leggibili i nomi delle feature
        content = content.replace("bow_", "")

        indent = "  " * max(0, depth - 1)
        parsed_lines.append(f"{indent}- {content}")

    return "\n".join(parsed_lines)

def verbalize_explanation(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Genera un dict di spiegazioni strutturate (JSON) per ogni emozione attiva,
    con campi:
      - simple_explanation : testo multilinea per l’utente non tecnico
      - emotion_words      : lista di parole emotive REALI trovate nel testo
      - parsed_rules       : regole surrogate “leggibili” (stringa indentata)
      - campi tecnici (probability, metrics, feature_importances, ecc.)
    """
    explanations: Dict[str, Any] = {}
    predictions    = result.get("predictions", {})   # {emo: score, …}
    features_list  = result.get("features", [])      # [ {feat: val, …}, … ]
    surrogates     = result.get("surrogates", {})    # { emo: {rules, metrics, importances}, … }
    original_text  = result.get("original_text", "")

    # Se esiste, prendo il primo feature vector (potrebbe avere n_emotion_words, stopword_ratio…)
    base_feats = features_list[0] if len(features_list) > 0 else {}

    for emo, data in surrogates.items():
        prob        = predictions.get(emo, 0.0)
        metrics     = data.get("metrics", {})
        importances = data.get("feature_importances", {})
        raw_rules   = data.get("rules", "").strip()

        # ————————————————————————————————————————————
        # 1) Trovo le prime 3 parole chiave (bow_<…>) con peso > 0
        active_indices = [
            (feat.replace("bow_", ""), imp)
            for feat, imp in importances.items()
            if feat.startswith("bow_") and imp > 0
        ]
        active_indices.sort(key=lambda x: x[1], reverse=True)
        top_keywords = [kw for kw, _ in active_indices[:3]]
        # ————————————————————————————————————————————

        # ————————————————————————————————————————————
        # 2) Conteggi reali dal testo (override a eventuali base_feats)
        word_count = count_words(original_text)
        n_excl     = count_exclamations(original_text)
        n_qst      = count_questions(original_text)
        # ————————————————————————————————————————————

        # ————————————————————————————————————————————
        # 3) Rapporto stop-word (se esiste in base_feats)
        stop_ratio = base_feats.get("stopword_ratio", None)
        stop_ratio_rounded = round(stop_ratio, 2) if stop_ratio is not None else None
        # ————————————————————————————————————————————

        # ————————————————————————————————————————————
        # 4) Estrazione di parole emotive REALI presenti nel testo
        emotion_words = extract_emotion_words(original_text, EMOTION_VOCAB)

        # Se non ne trovo e ho top_keywords, uso il fallback con la top_keywords[0]
        if not emotion_words and len(top_keywords) > 0:
            emotion_words = [top_keywords[0]]
        # ————————————————————————————————————————————

        # ————————————————————————————————————————————
        # 5) Costruisco la “spiegazione semplice” come elenco di frasi,
        #    separando ogni frase con un newline per maggiore leggibilità.
        frasi: List[str] = []
        emocapital = emo.upper()

        # a) Emozione + probabilità
        frasi.append(
            f"Il modello ha classificato il testo come “{emocapital}” (probabilità: {prob:.2f})."
        )

        # b) Parola chiave più rilevante (se esiste)
        if len(top_keywords) > 0:
            frasi.append(f"La parola chiave più rilevante è “{top_keywords[0]}”.")
        else:
            frasi.append("Non ci sono parole chiave particolarmente influenti.")

        # c) Conteggio parole totali
        frasi.append(
            f"Il testo è composto da {word_count} {'parola' if word_count == 1 else 'parole'}."
        )

        # d) Negazioni
        has_neg = bool(base_feats.get("has_negation", 0))
        if has_neg:
            frasi.append("Nel testo è presente almeno una negazione (‘non’).")

        # e) Punti esclamativi/interrogativi
        if n_excl > 0:
            frasi.append(
                f"Sono presenti {n_excl} {'punto esclamativo' if n_excl == 1 else 'punti esclamativi'}."
            )
        if n_qst > 0:
            frasi.append(
                f"Sono presenti {n_qst} {'punto interrogativo' if n_qst == 1 else 'punti interrogativi'}."
            )

        # f) Rapporto stop-word (se disponibile)
        if stop_ratio_rounded is not None:
            frasi.append(
                f"Il rapporto stop-word è {stop_ratio_rounded:.2f} "
                f"(rapporto tra parole neutre e parole emotive)."
            )

        # g) Parole emotive reali
        if emotion_words:
            frasi.append(f"Le parole emotive rilevate nel testo sono: {', '.join(emotion_words)}.")
        else:
            frasi.append("Nessuna parola emotiva rilevata.")

        # h) Conclusione sintetica
        if len(top_keywords) > 0:
            frasi.append(
                f"In sintesi, la presenza di “{top_keywords[0]}” e la lunghezza del testo "
                f"hanno portato il modello a predire “{emocapital}”."
            )
        else:
            frasi.append(
                f"In sintesi, segnali come lunghezza, stop-word e punteggiatura hanno portato "
                f"il modello a predire “{emocapital}”."
            )

        # Uso il newline per separare meglio le frasi nella semplice spiegazione
        simple_explanation = "\n".join(frasi)
        # ————————————————————————————————————————————

        # ————————————————————————————————————————————
        # 6) Costruisco la lista di tutte le feature_importances (feat, peso) con imp > 0
        all_imps = sorted(
            [
                (feat.replace("bow_", ""), round(imp, 4))
                for feat, imp in importances.items()
                if imp > 0
            ],
            key=lambda x: x[1],
            reverse=True
        )
        # ————————————————————————————————————————————

        # ————————————————————————————————————————————
        # 7) “Parsed rules”: se esiste raw_rules, lo converto in elenco indentato
        parsed = parse_rules(raw_rules) if raw_rules else ""
        # ————————————————————————————————————————————

        # ————————————————————————————————————————————
        # 8) Creo il dizionario che conterrà ogni dettaglio avanzato/tecnico
        explanations[emo] = {
            # 8.1) Spiegazione semplice (multilinea)
            "simple_explanation": simple_explanation,

            # 8.2) Parole emotive (dalla lista estratta + fallback)
            "emotion_words": emotion_words,

            # 8.3) Campi tecnici quantitativi
            "emotion": emo,
            "probability": round(prob, 2),
            "top_keywords": top_keywords,
            "word_count": word_count,
            "n_emotion_words": base_feats.get("n_emotion_words", None),
            "has_negation": has_neg,
            "stopword_ratio": stop_ratio_rounded,
            "exclamation_count": n_excl,
            "question_count": n_qst,

            # 8.4) Metriche surrogate
            "metrics": {
                "fidelity": round(metrics.get("fidelity", 0.0), 2),
                "sparsity": round(metrics.get("sparsity", 0.0), 2),
                "stability": round(metrics.get("stability", 0.0), 2),
            },

            # 8.5) Feature importances ordinate
            "all_importances": all_imps,

            # 8.6) Le regole surrogate “raw” e “parsed”
            "raw_rules": raw_rules,
            "parsed_rules": parsed,
        }
        # ————————————————————————————————————————————

    return explanations
