# download_lexicon.py

import os
import urllib.request

def download_nrc_lexicon():
    url = "https://nyc3.digitaloceanspaces.com/ml-files-distro/v1/upshot-trump-emolex/data/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt"
    dest_dir = os.path.join(os.path.dirname(__file__), "lexicon")
    os.makedirs(dest_dir, exist_ok=True)

    filename = "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
    dest_path = os.path.join(dest_dir, filename)

    if os.path.exists(dest_path):
        print(f"[✓] Il file esiste già: {dest_path}")
        return

    print(f"[↓] Downloading NRC Emotion Lexicon in: {dest_path}")
    urllib.request.urlretrieve(url, dest_path)
    print("[✓] Download completato.")

if __name__ == "__main__":
    download_nrc_lexicon()
