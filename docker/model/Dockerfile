FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install tools and Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /app

# Copy full project (or adjust for partial copy)
COPY ../.. .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Aggiungi questo comando per scaricare le risorse NLTK
RUN python -m nltk.downloader wordnet omw-1.4 averaged_perceptron_tagger averaged_perceptron_tagger_eng stopwords

# Default command: keep container alive for exec
CMD ["sleep", "infinity"]
