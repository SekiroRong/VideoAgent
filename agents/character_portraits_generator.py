import json
import os
from .utils import text2image, image2image
from .state import VideoGenState


prompt_template_front = \
"""
Generate a full-body, front-view portrait of character {identifier} based on the following description, with a pure white background. The character should be centered in the image, occupying most of the frame. Gazing straight ahead. Standing with arms relaxed at sides. Natural expression.
Features: {features}
Style: {style}
"""

prompt_template_side = \
"""
Generate a full-body, side-view portrait of character {identifier} based on the provided front-view portrait, with a pure white background. The character should be centered in the image, occupying most of the frame. Facing left. Standing with arms relaxed at sides.
"""

prompt_template_back = \
"""
Generate a full-body, back-view portrait of character {identifier} based on the provided front-view portrait, with a pure white background. The character should be centered in the image, occupying most of the frame. No facial features should be visible.
"""


def generate_character_images(state: VideoGenState) -> VideoGenState:
    image_save_root = os.path.join(state['cache_dir'], "character_portraits")
    os.makedirs(image_save_root, exist_ok=True)

    state["character_images"] = {}
    for character in state["character_desc"]:
        character_dir = os.path.join(image_save_root, f"{character.idx}_{character.identifier_in_scene}")
        os.makedirs(character_dir, exist_ok=True)

        front_portrait_path = os.path.join(character_dir, "front.png")
        side_portrait_path = os.path.join(character_dir, "side.png")
        back_portrait_path = os.path.join(character_dir, "back.png")
        if os.path.exists(front_portrait_path):
            pass
        else:
            features = "(static) " + character.static_features + "; (dynamic) " + character.dynamic_features
            prompt = prompt_template_front.format(
                identifier=character.identifier_in_scene,
                features=features,
                style=state["style"],
            )
            text2image(prompt, front_portrait_path)

        if os.path.exists(side_portrait_path):
            pass
        else:
            features = "(static) " + character.static_features + "; (dynamic) " + character.dynamic_features
            prompt = prompt_template_side.format(
                identifier=character.identifier_in_scene,
                features=features,
                style=state["style"],
            )
            image2image(prompt, [front_portrait_path], side_portrait_path)

        if os.path.exists(back_portrait_path):
            pass
        else:
            features = "(static) " + character.static_features + "; (dynamic) " + character.dynamic_features
            prompt = prompt_template_back.format(
                identifier=character.identifier_in_scene,
                features=features,
                style=state["style"],
            )
            image2image(prompt, [front_portrait_path], back_portrait_path)

        state["character_images"][character.identifier_in_scene] = {
                "front": {
                    "path": front_portrait_path,
                    "description": f"A front view portrait of {character.identifier_in_scene}.",
                },
                "side": {
                    "path": side_portrait_path,
                    "description": f"A side view portrait of {character.identifier_in_scene}.",
                },
                "back": {
                    "path": back_portrait_path,
                    "description": f"A back view portrait of {character.identifier_in_scene}.",
                },
            }

    return state