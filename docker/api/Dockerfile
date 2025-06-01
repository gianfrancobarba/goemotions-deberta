FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04


# Install tools and Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip


# Copia codice
WORKDIR /app
COPY ../.. .

# Installa dipendenze
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN python -m nltk.downloader wordnet omw-1.4 averaged_perceptron_tagger averaged_perceptron_tagger_eng stopwords

EXPOSE 8000
CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
