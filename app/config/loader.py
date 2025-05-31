import os
import yaml

class Config:
    def __init__(self, path: str = None):
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(path, "r") as f:
            self.cfg = yaml.safe_load(f)

    def get(self, section: str, key: str, default=None):
        return self.cfg.get(section, {}).get(key, default)

    def __getitem__(self, key):
        return self.cfg.get(key)

# Singleton che puoi importare ovunque
CFG = Config().cfg
