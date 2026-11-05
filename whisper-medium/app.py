import os
import tempfile
import time
import logging
import subprocess
import re
import gc
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import JSONResponse
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("whisperx-service")

# Auto-detect device and optimal compute type
if torch.cuda.is_available():
    DEVICE = "cuda"
    COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "float16")
    DEFAULT_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
else:
    DEVICE = "cpu"
    COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "int8")
    DEFAULT_MODEL = os.environ.get("WHISPER_MODEL", "base")

logger.info(f"Target execution environment: DEVICE={DEVICE}, COMPUTE_TYPE={COMPUTE_TYPE}, MODEL={DEFAULT_MODEL}")

class WhisperXTranscriber:
    def __init__(self):
        self.device = DEVICE
        self.compute_type = COMPUTE_TYPE
        self.model_name = DEFAULT_MODEL
        self.vad_options = {
            "vad_onset": 0.3,
            "vad_offset": 0.2,
        }
        self.align_models = {}
        self.diarize_model = None
        self.model = None

    def load_models(self):
        """Loads WhisperX models into memory."""
        import whisperx

        logger.info("Loading WhisperX transcription model (%s) on %s (%s)...", self.model_name, self.device, self.compute_type)
        self.model = whisperx.load_model(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            vad_options=self.vad_options,
        )

        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            logger.info("Loading Pyannote diarization pipeline...")
            from whisperx.diarize import DiarizationPipeline
            try:
                self.diarize_model = DiarizationPipeline(
                    model_name="pyannote/speaker-diarization-3.1",
                    token=hf_token,
                    device=self.device,
                )
            except Exception as e:
                logger.error("Failed to load DiarizationPipeline: %s. Falling back to transcription-only.", e)
                self.diarize_model = None
        else:
            logger.warning("HF_TOKEN env var not set. Diarization will be disabled.")
            self.diarize_model = None

    def _preprocess_audio(self, input_path: str, output_path: str) -> dict:
        vol_cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-af", "volumedetect",
            "-f", "null", "/dev/null"
        ]
        vol_res = subprocess.run(vol_cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        stderr_output = vol_res.stderr or ""

        mean_match = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", stderr_output)
        max_match = re.search(r"max_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", stderr_output)

        mean_vol = float(mean_match.group(1)) if mean_match else -20.0
        max_vol = float(max_match.group(1)) if max_match else 0.0

        if max_vol < -8.0 or mean_vol < -25.0:
            filter_chain = "highpass=f=80,loudnorm=I=-16:TP=-1.5:LRA=11"
        else:
            filter_chain = "highpass=f=80"

        conv_cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-af", filter_chain,
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            output_path
        ]

        try:
            subprocess.run(conv_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError:
            fallback_cmd = [
                "ffmpeg", "-y", "-i", input_path,
                "-ar", "16000",
                "-ac", "1",
                "-c:a", "pcm_s16le",
                output_path
            ]
            subprocess.run(fallback_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            filter_chain = "fallback_pcm_s16le"

        return {
            "max_volume_db": max_vol,
            "mean_volume_db": mean_vol,
            "filter_chain": filter_chain
        }

    def transcribe(self, audio_bytes: bytes, audio_suffix: str, language: str = "auto") -> dict:
        import whisperx

        processing_start = time.time()
        file_size_bytes = len(audio_bytes)
        tmp_raw_path = None
        tmp_proc_path = None

        try:
            with tempfile.NamedTemporaryFile(suffix=audio_suffix, delete=False) as tmp_raw:
                tmp_raw.write(audio_bytes)
                tmp_raw_path = tmp_raw.name

            proc_fd, tmp_proc_path = tempfile.mkstemp(suffix=".wav")
            os.close(proc_fd)

            self._preprocess_audio(tmp_raw_path, tmp_proc_path)

            audio = whisperx.load_audio(tmp_proc_path)
            audio_duration_sec = len(audio) / 16000.0

            asr_options = {}
            if language and language.strip() and language.strip().lower() != "auto":
                asr_options["language"] = language.strip()

            batch_size = 16 if self.device == "cuda" else 4
            result = self.model.transcribe(
                audio,
                batch_size=batch_size,
                vad_options=self.vad_options,
                **asr_options
            )
            detected_language = result.get("language", "en")

            # Word timestamp alignment
            is_aligned = False
            try:
                if detected_language not in self.align_models:
                    align_model, align_metadata = whisperx.load_align_model(
                        language_code=detected_language,
                        device=self.device,
                    )
                    self.align_models[detected_language] = (align_model, align_metadata)

                align_model, align_metadata = self.align_models[detected_language]
                result = whisperx.align(
                    result["segments"],
                    align_model,
                    align_metadata,
                    audio,
                    self.device,
                    return_char_alignments=False,
                )
                is_aligned = True
            except Exception as align_err:
                logger.error("Alignment failed for language '%s': %s", detected_language, align_err)

            # Diarization
            is_diarized = False
            if self.diarize_model is not None:
                try:
                    diarize_segments = self.diarize_model(tmp_proc_path)
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                    is_diarized = True
                except Exception as diarize_err:
                    logger.error("Diarization failed: %s", diarize_err)

            # Group consecutive speech by speaker
            raw_segments = result.get("segments", [])
            conversation = []
            unique_speakers = set()

            transcribed_start = None
            transcribed_end = None

            for seg in raw_segments:
                speaker = seg.get("speaker", "SPEAKER_00")
                text = seg.get("text", "").strip()
                unique_speakers.add(speaker)

                seg_start = seg.get("start")
                seg_end = seg.get("end")
                if seg_start is not None:
                    if transcribed_start is None or seg_start < transcribed_start:
                        transcribed_start = seg_start
                if seg_end is not None:
                    if transcribed_end is None or seg_end > transcribed_end:
                        transcribed_end = seg_end

                if not text:
                    continue

                if conversation and conversation[-1]["speaker"] == speaker:
                    conversation[-1]["text"] += " " + text
                else:
                    conversation.append({
                        "speaker": speaker,
                        "text": text
                    })

            processing_time_sec = round(time.time() - processing_start, 2)
            transcribed_duration = round(transcribed_end - transcribed_start, 2) if transcribed_start is not None and transcribed_end is not None else 0.0

            return {
                "language": detected_language,
                "transcript": conversation,
                "audio_meta": {
                    "file_size_bytes": file_size_bytes,
                    "audio_duration_sec": round(audio_duration_sec, 2),
                    "transcribed_duration_sec": transcribed_duration,
                    "raw_segment_count": len(raw_segments),
                    "merged_segment_count": len(conversation),
                    "speaker_count": len(unique_speakers),
                    "speakers": sorted(list(unique_speakers)),
                    "is_aligned": is_aligned,
                    "is_diarized": is_diarized,
                    "processing_time_sec": processing_time_sec,
                    "device": self.device,
                    "compute_type": self.compute_type,
                    "model": self.model_name
                },
            }

        finally:
            if tmp_raw_path and os.path.exists(tmp_raw_path):
                try:
                    os.unlink(tmp_raw_path)
                except Exception:
                    pass
            if tmp_proc_path and os.path.exists(tmp_proc_path):
                try:
                    os.unlink(tmp_proc_path)
                except Exception:
                    pass

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


app = FastAPI(title="WhisperX Transcription Service")
transcriber = WhisperXTranscriber()

@app.on_event("startup")
def startup_event():
    transcriber.load_models()

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "device": transcriber.device,
        "compute_type": transcriber.compute_type,
        "model": transcriber.model_name,
        "diarization_enabled": transcriber.diarize_model is not None,
    }

@app.post("/transcribe")
async def api_transcribe(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    authorization: Optional[str] = Header(None)
):
    secret_key = os.environ.get("TRANSCRIPTION_API_KEY")
    if secret_key and authorization != f"Bearer {secret_key}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    content = await file.read()
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"

    result = transcriber.transcribe(
        audio_bytes=content,
        audio_suffix=suffix,
        language=language
    )

    return JSONResponse(content=result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)

