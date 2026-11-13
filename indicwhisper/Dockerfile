FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    INDIC_MODEL_ID=ai4bharat/indicwav2vec-hindi

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    torch==2.1.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    transformers==4.44.2 \
    'numpy<2' \
    soundfile \
    pyctcdecode==0.5.0 \
    kenlm==0.3.0 \
    pyannote.audio==3.3.2

RUN python -c "import kenlm, pyctcdecode; from transformers import Wav2Vec2ProcessorWithLM"

COPY app.py /app/app.py

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
