from wajeezai_api.services.audio_processor import TranscriptionResult


class AudioPostProcessing:
    @staticmethod
    def post_process(audio_result : TranscriptionResult) -> TranscriptionResult:
        """
        Post-processes the audio transcription result to clean up the text and segments.
        """
        # Clean up the full transcription text
        return audio_result