from ast import Dict, List
from dataclasses import dataclass
from typing import Any

@dataclass
class NotKnowTheyTypeYet:
    pass
    
class Alignment:
    @staticmethod
    def align(audio_transcription:NotKnowTheyTypeYet, image_transcription: List[Dict[str, Any]]):
        pass