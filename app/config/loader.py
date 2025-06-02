import os
import yaml
from types import SimpleNamespace


def dict_to_namespace(d):
    """Ricorsivamente converte un dict in oggetto dot-accessibile."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d


# Percorso automatico del file config.yaml (nella stessa cartella di questo file)
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

# Caricamento YAML come dict puro
with open(CONFIG_PATH, "r") as f:
    raw_cfg = yaml.safe_load(f)

# 1. Dot-accesso (CFG.model.name)
CFG = dict_to_namespace(raw_cfg)

# 2. Accesso anche come dict (CFG_DICT["model"]["name"])
CFG_DICT = raw_cfg
