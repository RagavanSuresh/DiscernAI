from pydub import AudioSegment
import os

def convert_to_wav(audio_path, output_dir="processed_audio"):
    # Check if output directory exists, create if it doesn't
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        # Delete all files in the output directory
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    
    # Convert audio to .wav format
    audio = AudioSegment.from_file(audio_path)
    wav_path = os.path.join(output_dir, "processed_audio.wav")
    audio.export(wav_path, format="wav")
    
    return wav_path

# Example usage
audio_file = ".\dataset\Panel Discussion_ Are Young Students Getting Too Much Homework_.wav"  # Replace with your audio file path
output_wav = convert_to_wav(audio_file)
print(f"Audio converted and saved as: {output_wav}")
