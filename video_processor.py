from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import os

def extract_audio(video_path, output_dir="processed_audio"):
    # Check if output directory exists, create if it doesn't
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        # Delete all files in the output directory
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    
    # Load the video
    try:
        video = VideoFileClip(video_path)
    except Exception as e:
        raise ValueError(f"Error loading video file: {e}")
    
    # Extract the audio
    temp_audio_path = os.path.join(output_dir, "temp_audio.mp3")
    video.audio.write_audiofile(temp_audio_path)
    
    # Convert audio to .wav format with a fixed name
    audio = AudioSegment.from_file(temp_audio_path)
    wav_path = os.path.join(output_dir, "processed_audio.wav")
    audio.export(wav_path, format="wav")
    
    # Cleanup the intermediate .mp3 file
    os.remove(temp_audio_path)
    
    return wav_path

# Example usage
video_file = ".\\dataset\\Panel discussion_ Making metro areas livable.mp4"  # Replace with your video file path
output_wav = extract_audio(video_file)
print(f"Audio extracted and saved as: {output_wav}")
