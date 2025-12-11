import json
import os
import logging
from .utils import sample_call_i2v
from .state import VideoGenState


def generate_single_video(state: VideoGenState) -> VideoGenState:
    for idx, shots in enumerate(state["shot_descriptions"]):
        for j, shot_description in enumerate(shots):
            video_path = os.path.join(state['cache_dir'], f"scene_{idx}", f"shot_{j}", "video.mp4")
            if os.path.exists(video_path):
                logging.info(f"🚀 Skipped generating video for shot {shot_description.idx}, already exists.")
                pass
            else:
                frame_paths = []
                frame_paths.append(os.path.join(state['cache_dir'], f"scene_{idx}", f"shot_{j}", "first_frame.png"))
                last_frame_path = os.path.join(state['cache_dir'], f"scene_{idx}", f"shot_{j}", "last_frame.png")
                if os.path.exists(last_frame_path):
                    frame_paths.append(last_frame_path)
                prompt=shot_description.motion_desc + "\n" + shot_description.audio_desc
                logging.info(f"☑️ Generated video for shot {shot_description.idx}, saved to {video_path}.")

    return state