import os
import json
from datetime import datetime, timezone
from config.loader import CFG  # Assicurati che sia accessibile correttamente

LOG_DIR = CFG.paths.api_logs
os.makedirs(LOG_DIR, exist_ok=True)

def log_request(route: str, input_text: str, output_data: dict):
    print("Logging to:", LOG_DIR)
    log_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "route": route,
        "input": input_text,
        "output": output_data
    }

    filename = os.path.join(LOG_DIR, f"log_{datetime.now(timezone.utc).date()}.jsonl")
    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_data) + "\n")
