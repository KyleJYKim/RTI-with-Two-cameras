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

def compute_delay(static_audio, moving_audio):
    # compute audio delay using cross-correlation.
    static_y, static_sr = librosa.load(static_audio, sr=8000)   # reference audio
    moving_y, moving_sr = librosa.load(moving_audio, sr=8000)
    
    static_y = static_y[:10 * static_sr]
    moving_y = moving_y[:10 * moving_sr]
    
    assert static_sr == moving_sr, "Sample rates must match"
    
    corr = np.correlate(static_y, moving_y, mode='full')
    lag = np.argmax(corr) - (len(moving_y) - 1)
    delay_sec = lag / static_sr
    
    print(f"Sync delay estimated: {delay_sec:.4f} seconds")
    
    return delay_sec

def shift_and_save_moving_video(org_video, shfd_video, delay_sec):
    # shift video
    (
        ffmpeg.input(org_video, itsoffset=delay_sec)     # shift the audio & video
        .output(shfd_video, vcodec='copy', acodec='aac')
        .overwrite_output()
        .run()
    )
    print("Moving video shifted and saved")

def main():
    print("Start Video Synchronization")
    
    extract_audio("./data/static/coin.MOV", "./data/static/coin_audio.wav")  # static video = reference video
    extract_audio("./data/moving/coin.MOV", "./data/moving/coin_audio.wav")  # moving video = to be shifted

    delay = compute_delay("./data/static/coin_audio.wav", "./data/moving/coin_audio.wav")
    shift_and_save_moving_video("./data/moving/coin.MOV", "./data/moving/coin_shifted.MOV", delay)
    
    print("End Video Synchronization")
    
if __name__ == "__main__":
    main()