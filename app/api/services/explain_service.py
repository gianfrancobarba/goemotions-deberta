# File: app/api/services/explain_service.py

from model.explainability.explain_pipeline import explain as explain_pipeline

def explain_emotions(text: str) -> dict:
    """
    Invoca la pipeline di explainability e restituisce tutto il JSON generato.
    """
    return explain_pipeline(text)
