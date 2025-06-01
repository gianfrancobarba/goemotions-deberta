# test_predictor.py
from model.explainability.predictor import predict_all
import json

if __name__ == "__main__":
    inputs = ["I am happy!", "I feel sad."]
    outputs = predict_all(inputs)
    print(json.dumps(outputs, indent=2))
