FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /app/model

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader wordnet omw-1.4 averaged_perceptron_tagger averaged_perceptron_tagger_eng stopwords


CMD ["python"]
