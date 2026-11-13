import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch


class DecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fastapi = ModuleType("fastapi")
        for name in ("FastAPI", "UploadFile", "File", "Form", "Header", "HTTPException"):
            setattr(fastapi, name, MagicMock())
        responses = ModuleType("fastapi.responses")
        responses.JSONResponse = MagicMock()
        cls.torch = MagicMock()
        cls.torch.cuda.is_available.return_value = False
        transformers = ModuleType("transformers")

        class LMProcessor:
            def __call__(self, *args, **kwargs):
                return SimpleNamespace(input_values=MagicMock())

        cls.LMProcessor = LMProcessor
        transformers.Wav2Vec2ProcessorWithLM = LMProcessor
        cls.modules = patch.dict(sys.modules, {
            "fastapi": fastapi,
            "fastapi.responses": responses,
            "torch": cls.torch,
            "torchaudio": MagicMock(),
            "transformers": transformers,
        })
        cls.modules.start()
        spec = importlib.util.spec_from_file_location("indicwhisper_app", Path(__file__).with_name("app.py"))
        cls.app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.app)

    @classmethod
    def tearDownClass(cls):
        cls.modules.stop()

    def test_language_model_decodes_cpu_logits(self):
        processor = self.LMProcessor()
        processor.batch_decode = MagicMock(return_value=SimpleNamespace(text=[" decoded "]))
        logits = MagicMock()
        transcriber = self.app.IndicTranscriber()
        transcriber.processor = processor

        self.assertEqual(transcriber._decode_logits(logits), "decoded")
        processor.batch_decode.assert_called_once_with(logits.cpu.return_value.numpy.return_value)

    def test_standard_processor_decodes_token_ids(self):
        processor = MagicMock()
        processor.batch_decode.return_value = [" decoded "]
        logits = MagicMock()
        transcriber = self.app.IndicTranscriber()
        transcriber.processor = processor

        self.assertEqual(transcriber._decode_logits(logits), "decoded")
        self.torch.argmax.assert_called_with(logits, dim=-1)
        processor.batch_decode.assert_called_once_with(self.torch.argmax.return_value)

    def test_transcribe_uses_language_model_decoder(self):
        processor = self.LMProcessor()
        processor.batch_decode = MagicMock(return_value=SimpleNamespace(text=[" hello "]))
        waveform = MagicMock()
        waveform.shape = (1, 3200)
        waveform.__getitem__.return_value.shape = (1, 3200)
        logits = MagicMock()
        transcriber = self.app.IndicTranscriber()
        transcriber.processor = processor
        transcriber.model = MagicMock(return_value=SimpleNamespace(logits=logits))

        with patch.object(transcriber, "_convert_audio_16k"), \
                patch.object(self.app.torchaudio, "load", return_value=(waveform, 16000)):
            result = transcriber.transcribe(b"audio", ".wav")

        self.assertEqual(result["transcript"], [{"speaker": "SPEAKER_00", "text": "hello"}])
        processor.batch_decode.assert_called_once_with(logits.cpu.return_value.numpy.return_value)


if __name__ == "__main__":
    unittest.main()
