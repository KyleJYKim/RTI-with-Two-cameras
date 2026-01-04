#!/usr/bin/env python3
import ffmpeg
import librosa
import numpy as np
import cv2 as cv

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

def trim_video(input_video, output_video, start_time, target_fps=None):
    input = ffmpeg.input(input_video, ss=start_time)

    if target_fps is not None:
        input = input.filter("fps", fps=target_fps)

    (
        input
        .output(output_video)
        .overwrite_output()
        .run()
    )

    if target_fps is None:
        print("Video trimmed and saved")
    else:
        print(f"Video trimmed, resampled to {target_fps} fps, and saved")
    
def main():
    print("Start Video Synchronization")
    
    static_video = "./data_G/cam1/coin1.mov"
    moving_video = "./data_G/cam2/coin1.mp4"
    static_sound = "./data_G/cam1/coin1_audio.wav"
    moving_sound = "./data_G/cam2/coin1_audio.wav"
    static_synced = "./data_G/cam1/coin1_synced.mov"
    moving_synced = "./data_G/cam2/coin1_synced.mov"
    
    vid1 = cv.VideoCapture(static_video)
    vid2 = cv.VideoCapture(moving_video)
    fps1 = vid1.get(cv.CAP_PROP_FPS)
    fps2 = vid2.get(cv.CAP_PROP_FPS)
    print(fps1, fps2)
    
    extract_audio(static_video, static_sound)  # static video = reference video
    extract_audio(moving_video, moving_sound)  # moving video = to be shifted
    
    t_static = detect_cue_time(static_sound)
    t_moving = detect_cue_time(moving_sound)
    
    trim_video(static_video, static_synced, t_static, fps1)
    trim_video(moving_video, moving_synced, t_moving, fps1)
    
    print("End Video Synchronization")
    
if __name__ == "__main__":
    main()