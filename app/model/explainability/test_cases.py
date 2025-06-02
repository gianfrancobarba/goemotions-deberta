# test_cases.py

"""
Test automatici dei moduli explainability:
- perturber.py
- predictor.py
- features.py
- surrogate.py
- metrics.py
"""

import os
import torch

from perturber import generate_perturbations
from predictor import predict_all
from features import extract_features
from surrogate import explain_with_surrogates
from config_surrogate import THRESHOLD


def test_perturber():
    text = "I'm very happy today!"
    perturbed = generate_perturbations(text, n=5)
    assert isinstance(perturbed, list) and len(perturbed) == 5
    print("test_perturber passed")


def test_predictor():
    texts = ["I'm happy.", "I'm sad."]
    results = predict_all(texts)
    assert isinstance(results, list) and len(results) == 2
    assert all(isinstance(r, dict) for r in results)
    print("test_predictor passed")


def test_features():
    texts = ["I'm excited!", "No one listens to me."]
    df = extract_features(texts)
    assert df.shape[0] == 2
    assert any(col.startswith("bow_") for col in df.columns)
    print("test_features passed")


def test_surrogate():
    text = "I finally did it, I'm so proud of myself!"
    result = explain_with_surrogates(text)
    assert isinstance(result, dict) and len(result) > 0
    for info in result.values():
        assert "tree_rules" in info and "metrics" in info
    print("test_surrogate passed")


def run_all_tests():
    print("\nRunning tests for explainability modules:\n")
    test_perturber()
    test_predictor()
    test_features()
    test_surrogate()
    print("\nAll tests passed!")


if __name__ == "__main__":
    run_all_tests()