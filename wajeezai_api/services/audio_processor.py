import librosa

class AudioProcessor:

    @staticmethod
    def transcribe_long_audio_whisper_native(model, processor, audio_path, device="cuda"):
        """
        Uses Whisper's built-in long-form generation + timestamps.
        No manual chunking needed — returns segments with (start, end, text) directly usable in fusion.
        """

        audio, sr = librosa.load(audio_path, sr=16000)

        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)

        predicted_ids = model.generate(
            input_features,
            return_timestamps=True,
            language="ar",
            task="transcribe",
        )

        result = processor.batch_decode(
            predicted_ids, skip_special_tokens=True, output_offsets=True
        )

        segments = []
        for offset in result[0]["offsets"]:
            segments.append({
                "text": offset["text"].strip(),
                "start": offset["timestamp"][0],
                "end": offset["timestamp"][1],
            })

        full_transcription = " ".join(s["text"] for s in segments)
        return full_transcription, segments