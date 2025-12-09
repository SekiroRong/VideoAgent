import json
import os
from utils import any2video



def generate_scene_video(state: VideoGenState) -> VideoGenState:
    # 1. 获取保存根路径（优先使用state中的配置，无则用默认值）
    save_root = state.get("video_save_root", "./output/video")

    scene_video_path = os.path.join(save_root, state["video_name"] + '.mp4')
    any2video(state["scene_desc"], scene_video_path)
    state["scene_video"] = scene_video_path
    print(f"生成的场景视频片段：{scene_video_path}\n")
    return state