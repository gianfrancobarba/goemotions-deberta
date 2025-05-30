FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04


# Installazioni base
RUN apt-get update && apt-get install -y python3 python3-pip git

# Copia codice
WORKDIR /app
COPY ../.. .

# Installa dipendenze
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
