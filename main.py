import numpy as np
from moviepy import ImageSequenceClip, AudioFileClip

from audio_processing import process_audio_file, save_processed_audio, load_processed_audio
from audio_post_processing import flatten, normalize_heights, ballistic_smoothing
from video_generation import compile_frame_arrays, VisualizerTypes


VIDEO_RES = (1920, 1080)
H_PERC = 1
V_PERC = 0.3


if __name__ == "__main__":
    audio_framerate = 60
    video_framerate = 60

    processed_audio, frequency_bins, total_time = process_audio_file("test2.wav", bar_count=108, framerate=audio_framerate)


    smoothed_processed = normalize_heights(ballistic_smoothing(flatten(processed_audio, alpha=2/3), attack_time=0.005, decay_time=0.20, fps=audio_framerate))


    frame_arrs = compile_frame_arrays(VisualizerTypes.ROUNDED_BARS, smoothed_processed, int(VIDEO_RES[1] * V_PERC), int(VIDEO_RES[0] * H_PERC),
                                      np.array([0, 255, 0]), np.array([0, 0, 0]), width=10, bidirectional=True, corner_radius=1)

    audio_clip = AudioFileClip("test2.wav")
    clip = ImageSequenceClip(list(frame_arrs), fps=video_framerate).with_audio(audio_clip)
    clip.write_videofile("results/audio_visualizer.mp4", fps=video_framerate)