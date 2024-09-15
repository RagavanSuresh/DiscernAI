import whisper
import torch
from pyannote.audio import Pipeline
from pydub import AudioSegment
import pandas as pd

# Load Hugging Face token from file
with open('hugging face token.txt', 'r') as f:
    hf_token = f.read().strip()

# Initialize Whisper model (base model)
model = whisper.load_model("base")

# Set path to your WAV audio file
audio_wav_path = "./processed_audio/processed_audio.wav"

# Load the WAV audio file using pydub
audio = AudioSegment.from_wav(audio_wav_path)

# Initialize Pyannote Speaker Diarization pipeline
pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', use_auth_token=hf_token)

# Apply speaker diarization using the Pyannote pipeline
diarization = pipeline({"uri": "audio", "audio": audio_wav_path})

# Create a list to store speaker-wise transcriptions
transcriptions = []

# Open a file to write the transcriptions
with open('transcription.txt', 'w') as file:
    
    # Process each speech segment for each speaker identified by the diarization pipeline
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_time = turn.start
        end_time = turn.end
        duration = end_time - start_time

        # Extract the specific speaker's segment from the audio
        speaker_audio = audio[start_time * 1000:end_time * 1000]  # Convert seconds to milliseconds
        speaker_audio.export('temp.wav', format='wav')  # Save speaker's segment to temporary WAV file

        # Transcribe the speaker's segment using Whisper
        result = model.transcribe('temp.wav')

        # Append the transcription along with the speaker label, start time, end time, and text
        transcriptions.append({
            'speaker': speaker,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'text': result['text']
        })

        # Write the speaker's transcription to the file
        file.write(f"Speaker {speaker} ({start_time:.2f}s - {end_time:.2f}s): {result['text']}\n")

# Detect language of the full audio using Whisper
language_result = model.transcribe(audio_wav_path, task="detect-language")
language = language_result['language']

# Write the detected language to a separate file
with open('detected_language.txt', 'w') as lang_file:
    lang_file.write(f"Detected Language: {language}\n")

# Create a pandas DataFrame from the transcriptions
df = pd.DataFrame([{
    'speaker': t['speaker'],
    'duration': t['duration'],
    'document': t['text']
} for t in transcriptions])

# Save the DataFrame to a CSV file
df.to_csv('sample.csv')

# Display the pandas DataFrame
print("\nPandas DataFrame Output:")
print(df)
