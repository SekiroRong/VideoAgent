import json
import os
from utils import text2image



def generate_character_images(state: VideoGenState) -> VideoGenState:
    # 1. 获取保存根路径（优先使用state中的配置，无则用默认值）
    save_root = state.get("image_save_root", "./output/characters")

    char_image_path = os.path.join(save_root, state["character_name"] + '.png')
    text2image(state["character_desc"], char_image_path)
    state["character_images"] = char_image_path
    print(f"生成的人物图像：{char_image_path}\n")
    return state