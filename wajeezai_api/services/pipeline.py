from abc import ABC, abstractmethod

class AbstractPipeline(ABC):
    def __init__(self,audio_processor, image_processor, fuser):
        self.audio_processor = audio_processor
        self.image_processor = image_processor
        self.fuser = fuser



    @abstractmethod
    def process_input(self, audio , images) : # -> should save word document and return the status of the process
        """Process a payment. Must return True if successful."""
        pass

    @abstractmethod
    def process_audio(self, audio) -> str:
        """Process audio input. Must return str if successful."""
        pass

    @abstractmethod
    def process_images(self, images) -> str:   
        """Process image input. Must return str if successful."""
        pass
