import pickle
from pathlib import Path
from wajeezai_api.services.alignment import Alignment
from wajeezai_api.services.audio_processor import TranscriptionResult, TranscriptionSegment
from wajeezai_api.services.image_processor import SlideResult

cache_dir = Path(
    r"E:\Projects\WajeezAI\Test_Samples\sample_1\cache"
)

# Load transcription result
with open(cache_dir / "transcription_result.pkl", "rb") as f:
    transcription_result : TranscriptionResult = pickle.load(f) 

# Load parsed image output
with open(cache_dir / "parsed_output_with_meta.pkl", "rb") as f:
    parsed_output_with_meta: list[SlideResult] = pickle.load(f)

# Load parsed image output with timestamps
with open(cache_dir / "parsed_output_with_meta_with_timestamp.pkl", "rb") as f:
    parsed_output_with_meta_with_timestamp: list[SlideResult] = pickle.load(f)
print("parsed_output_with_meta_with_timestamp")
print(parsed_output_with_meta_with_timestamp)
segments = [
    TranscriptionSegment(text=s["text"], start=s["start"], end=s["end"])
    for s in transcription_result.segments  # whatever your Whisper pipeline output is called
]
transcription_result.segments = segments

alignment_result = Alignment.align(
    slides=parsed_output_with_meta_with_timestamp,
    transcription=transcription_result,
    text_threshold=0.30,
    visual_threshold=0.30
    )
print(f"alignment_result: {alignment_result}")