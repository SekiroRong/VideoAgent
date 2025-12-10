import json
import os
from .utils import sample_call_i2v
from .state import VideoGenState


def generate_single_video(state: VideoGenState) -> VideoGenState:

    for idx, shots in enumerate(state["shot_descriptions"]):
        for j, shot_description in enumerate(shots):
            video_path = os.path.join(state['cache_dir'], f"scene_{idx}", f"shot_{j}", "video.mp4")
            frame_paths = []
            frame_paths.append(os.path.join(state['cache_dir'], f"scene_{idx}", f"shot_{j}", "first_frame.png"))
            prompt=shot_description.motion_desc + "\n" + shot_description.audio_desc
            print(frame_paths, prompt)
            print('\n')

    return state