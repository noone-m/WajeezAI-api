from wajeezai_api.services.audio_processor import AudioProcessor

transcription_result = AudioProcessor.transcribe_long_audio_whisper_native(audio_path=r"E:\Projects\WajeezAI\Test_Samples\sample_1\ring_tree_topology_120s.wav")
print("full_transcription")
print(transcription_result.full_transcription)
print("segments")
print(transcription_result.segments) 


