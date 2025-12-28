#!/usr/bin/env python3
import ffmpeg
import librosa
import numpy as np

def extract_audio(video_path, wav_path):
    # extract mono WAV audio at 44.1kHz using ffmpeg-python
    (
        ffmpeg.input(video_path)
        .output(wav_path, acodec='pcm_s16le', ac=1, ar='44100')
        .overwrite_output()
        .run()
    )
    print("Audio extracted")

def detect_cue_time(wav_path, sr=8000, threshold=0.2):
    y, sr = librosa.load(wav_path, sr=sr)
    
    # RMS energy
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
    
    # first strong energy spike
    idx = np.argmax(rms > threshold * np.max(rms))
    return times[idx]

def trim_video(input_video, output_video, start_time):
    (
        ffmpeg
        .input(input_video, ss=start_time)
        .output(output_video)
        .overwrite_output()
        .run()
    )
    print("Moving video shifted and saved")
    
# def compute_delay(static_audio, moving_audio):
#     # compute audio delay using cross-correlation.
#     static_y, static_sr = librosa.load(static_audio, sr=8000)   # reference audio
#     moving_y, moving_sr = librosa.load(moving_audio, sr=8000)
    
#     static_y = static_y[:10 * static_sr]
#     moving_y = moving_y[:10 * moving_sr]
    
#     assert static_sr == moving_sr, "Sample rates must match"
    
#     corr = np.correlate(static_y, moving_y, mode='full')
#     lag = np.argmax(corr) - (len(moving_y) - 1)
#     delay_sec = lag / static_sr
    
#     print(f"Sync delay estimated: {delay_sec:.4f} seconds")
    
#     return delay_sec


def main():
    print("Start Video Synchronization")
    
    static_video = "./data_G/cam1/coin1.mov"
    moving_video = "./data_G/cam2/coin1.mp4"
    static_sound = "./data_G/cam1/coin1_audio.wav"
    moving_sound = "./data_G/cam2/coin1_audio.wav"
    static_synced = "./data_G/cam1/coin1_synced.mov"
    moving_synced = "./data_G/cam2/coin1_synced.mov"
    
    extract_audio(static_video, static_sound)  # static video = reference video
    extract_audio(moving_video, moving_sound)  # moving video = to be shifted
    
    t_static = detect_cue_time(static_sound)
    t_moving = detect_cue_time(moving_sound)
    
    trim_video(static_video, static_synced, t_static)
    trim_video(moving_video, moving_synced, t_moving)
    
    print("End Video Synchronization")
    
if __name__ == "__main__":
    main()