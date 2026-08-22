from dataclasses import dataclass
from pathlib import Path
from transformers import pipeline
import torch

MODEL_PATH = 'whisper-model-large'
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f'device is {device}')

asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=str(MODEL_PATH),
    device=0 if device == "cuda" else -1,
    chunk_length_s=30,       # internal chunking window
    stride_length_s=5,       # overlap between chunks, reduces boundary cut-offs
    return_timestamps=True,
)

@dataclass
class TranscriptionSegment:
    text: str
    start: float
    end: float


@dataclass
class TranscriptionResult:
    full_transcription: str
    segments: list[TranscriptionSegment]
    

class AudioProcessor:
    @staticmethod
    def transcribe_long_audio_whisper_native(audio_path=None, pipeline_obj=asr_pipeline):
        """
        Long-form Whisper transcription with segment timestamps, via the
        pipeline API (handles chunking + stitching internally — avoids the
        raw generate()/batch_decode() nested-output bug for audio >30s).
        """
        result = pipeline_obj(audio_path)

        segments = []
        for chunk in result["chunks"]:
            start, end = chunk["timestamp"]
            segments.append({
                "text": chunk["text"].strip(),
                "start": start,
                "end": end,
            })

        full_transcription = result["text"].strip()
        return full_transcription, segments

######## example_returned 
# full_transcription
# تمام واضح شو دور الترمينيتر هاي الترمينيتر طيب هلأ انا بدي اروح على الشكل الآتي بالرينك تبقى لو اجي لاحظ شايفين هلأ هذا الشكل الموجود معي انا هون هلأ انا بدي اشيل هذا الترمينيتر مع هذا الترمينيتر وبدي اربطه لهذا بدي اربطه نهاية هذا مع البداية هذا ما
# segments
# [{'text': 'تمام واضح شو دور الترمينيتر هاي الترمينيتر طيب هلأ انا بدي اروح على الشكل الآتي بالرينك تبقى لو اجي لاحظ شايفين هلأ هذا الشكل الموجود معي انا هون هلأ انا بدي اشيل هذا الترمينيتر مع هذا الترمينيتر وبدي اربطه لهذا بدي اربطه نهاية هذا مع البداية هذا ما', 'start': 0.0, 'end': 20.0}]