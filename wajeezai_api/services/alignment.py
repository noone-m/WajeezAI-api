from ast import Dict, List
from dataclasses import dataclass
from typing import Any
from audio_processor import TranscriptionResult as AudioTranscription
from image_processor import ImageInput, SlideResult

class Alignment:
    @staticmethod
    def align(audio_transcription: AudioTranscription, image_transcription: SlideResult):
        # align each image transcription with top k audio segments based on temporal and semantic similarity
        pass