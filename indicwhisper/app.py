import os
import tempfile
import time
import logging
import subprocess
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import JSONResponse
import torch
import torchaudio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("indicwhisper-service")

MODEL_ID = os.environ.get("INDIC_MODEL_ID", "ai4bharat/indicwav2vec-hindi")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class IndicTranscriber:
    def __init__(self):
        self.device = DEVICE
        self.model = None
        self.processor = None
        self.diarize_model = None

    def load_models(self):
        logger.info(f"Loading Indic ASR model: {MODEL_ID} on {self.device}...")
        try:
            from transformers import AutoModelForCTC, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(MODEL_ID)
            self.model = AutoModelForCTC.from_pretrained(MODEL_ID).to(self.device)
            self.model.eval()
            logger.info("Indic ASR model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Indic model: {e}")

        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            logger.info("Loading Pyannote Diarization for Indic pipeline...")
            try:
                from pyannote.audio import Pipeline
                self.diarize_model = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token
                )
                if self.device == "cuda":
                    self.diarize_model.to(torch.device("cuda"))
            except Exception as de:
                logger.warning(f"Pyannote diarization setup warning: {de}")
                self.diarize_model = None

    def _convert_audio_16k(self, in_path: str, out_path: str):
        cmd = ["ffmpeg", "-y", "-i", in_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", out_path]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    def transcribe(self, audio_bytes: bytes, audio_suffix: str) -> dict:
        processing_start = time.time()
        file_size_bytes = len(audio_bytes)

        with tempfile.NamedTemporaryFile(suffix=audio_suffix, delete=False) as raw_f:
            raw_f.write(audio_bytes)
            raw_path = raw_f.name

        proc_fd, proc_path = tempfile.mkstemp(suffix=".wav")
        os.close(proc_fd)

        try:
            self._convert_audio_16k(raw_path, proc_path)
            waveform, sample_rate = torchaudio.load(proc_path)
            audio_duration_sec = waveform.shape[1] / sample_rate

            # 1. Run Diarization if available
            segments = []
            if self.diarize_model:
                try:
                    diarization = self.diarize_model(proc_path, min_speakers=2, max_speakers=2)
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        segments.append({
                            "speaker": speaker,
                            "start": turn.start,
                            "end": turn.end
                        })
                except Exception as de:
                    logger.warning(f"Diarization failed: {de}")

            if not segments:
                segments = [{"speaker": "SPEAKER_00", "start": 0.0, "end": audio_duration_sec}]

            # 2. Transcribe segments using Indic Wav2Vec
            conversation = []
            unique_speakers = set()

            for seg in segments:
                start_sample = int(seg["start"] * sample_rate)
                end_sample = min(int(seg["end"] * sample_rate), waveform.shape[1])
                seg_wave = waveform[:, start_sample:end_sample]

                if seg_wave.shape[1] < 1600: # skip under 0.1s
                    continue

                inputs = self.processor(seg_wave.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
                with torch.no_grad():
                    logits = self.model(inputs).logits
                pred_ids = torch.argmax(logits, dim=-1)
                transcription = self.processor.batch_decode(pred_ids)[0].strip()

                if not transcription:
                    continue

                speaker = seg["speaker"]
                unique_speakers.add(speaker)

                if conversation and conversation[-1]["speaker"] == speaker:
                    conversation[-1]["text"] += " " + transcription
                else:
                    conversation.append({"speaker": speaker, "text": transcription})

            return {
                "language": "hi",
                "transcript": conversation,
                "audio_meta": {
                    "file_size_bytes": file_size_bytes,
                    "audio_duration_sec": round(audio_duration_sec, 2),
                    "speaker_count": len(unique_speakers),
                    "speakers": sorted(list(unique_speakers)),
                    "processing_time_sec": round(time.time() - processing_start, 2),
                    "model": MODEL_ID,
                    "device": self.device,
                }
            }
        finally:
            if os.path.exists(raw_path):
                os.unlink(raw_path)
            if os.path.exists(proc_path):
                os.unlink(proc_path)


app = FastAPI(title="AI4Bharat Indic ASR Service")
transcriber = IndicTranscriber()

@app.on_event("startup")
def startup():
    transcriber.load_models()

@app.get("/health")
def health():
    return {"status": "healthy", "model": MODEL_ID, "device": transcriber.device}

@app.post("/transcribe")
async def api_transcribe(file: UploadFile = File(...), authorization: Optional[str] = Header(None)):
    secret_key = os.environ.get("TRANSCRIPTION_API_KEY")
    if secret_key and authorization != f"Bearer {secret_key}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    content = await file.read()
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    result = transcriber.transcribe(content, suffix)
    return JSONResponse(content=result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)

