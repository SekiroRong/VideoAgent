import json
import os
import logging
from moviepy import VideoFileClip, concatenate_videoclips
from .state import VideoGenState


def merge_final_video(state: VideoGenState) -> VideoGenState:
    all_video_paths = []
    final_video_path = os.path.join(state['cache_dir'], "final_video.mp4")
    if os.path.exists(final_video_path):
        logging.info(f"🚀 Skipped concatenating videos, already exists.")
    else:
        logging.info(f"🎬 Starting concatenating videos...")
        for idx, shots in enumerate(state["shot_descriptions"]):
            for j, shot_description in enumerate(shots):
                video_path = os.path.join(state['cache_dir'], f"scene_{idx}", f"shot_{j}", "video.mp4")
                assert os.path.exists(video_path)
                all_video_paths.append(video_path)

        video_clips = [VideoFileClip(final_video_path)
                       for final_video_path in all_video_paths]
        final_video = concatenate_videoclips(video_clips)
        final_video.write_videofile(final_video_path)
        logging.info(f"☑️ Concatenated videos, saved to {final_video_path}.")

    return state