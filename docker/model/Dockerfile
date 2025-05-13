FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /app/model

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python"]
