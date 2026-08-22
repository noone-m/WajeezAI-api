from wajeezai_api.services.audio_processor import AudioProcessor

full_transcription, segments = AudioProcessor.transcribe_long_audio_whisper_native(audio_path=r"E:\Projects\WajeezAI\Test_Samples\sample_1\ring_tree_topology_120s.wav")
print("full_transcription")
print(full_transcription)
print("segments")
print(segments) 