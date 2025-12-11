import getpass
import os
import logging

if not os.getenv("DASHSCOPE_API_KEY"):
    print('Please set your API_KEY')

# Note: Domestic Chinese users should configure DASHSCOPE_API_BASE to the domestic endpoint, as langchain-qwq defaults to the international Alibaba Cloud endpoint.
if not os.getenv("DASHSCOPE_API_BASE"):
    os.environ["DASHSCOPE_API_BASE"] = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

from langchain_qwq import ChatQwen
from langchain.tools import tool
from .state import VideoGenState
from .utils import encode_file, image2image

model = ChatQwen(
    model="qwen3-vl-flash",
    # other params...
)

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict
import json

from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Tuple

from .interfaces import ShotBriefDescription

system_prompt_template_select_reference_images_only_text = \
"""
[Role]
You are a professional visual creation assistant skilled in multimodal image analysis and reasoning.

[Task]
Your core task is to intelligently select the most suitable reference images from a provided set of reference image descriptions (including multiple character reference images and existing scene images from prior frames) based on the user's text description (describing the target frame), ensuring that the subsequently generated image meets the following key consistencies:
- Character Consistency: The appearance (e.g. gender, ethnicity, age, facial features, hairstyle, body shape), clothing, expression, posture, etc., of the generated character should highly match the reference image descriptions.
- Environmental Consistency: The scene of the generated image (e.g., background, lighting, atmosphere, layout) should remain coherent with the existing image descriptions from prior frames.
- Style Consistency: The visual style of the generated image (e.g., realistic, cartoon, film-like, color tone) should harmonize with the reference image descriptions.

[Input]
You will receive a text description of the target frame, along with a sequence of reference image descriptions.
- The text description of the target frame is enclosed within <FRAME_DESC> and </FRAME_DESC>.
- The sequence of reference image descriptions is enclosed within <SEQ_DESC> and </SEQ_DESC>. Each description is prefixed with its index, starting from 0.

Below is an example of the input format:
<FRAME_DESC>
[Camera 1] Shot from Alice's over-the-shoulder perspective. Alice is on the side closer to the camera, with only her shoulder appearing in the lower left corner of the frame. Bob is on the side farther from the camera, positioned slightly right of center in the frame. Bob's expression shifts from surprise to delight as he recognizes Alice.
</FRAME_DESC>

<SEQ_DESC>
Image 0: A front-view portrait of Alice.
Image 1: A front-view portrait of Bob.
Image 2: [Camera 0] Medium shot of the supermarket aisle. Alice and Bob are shown in profile facing the right side of the frame. Bob is on the right side of the frame, and Alice is on the left side. Alice, looking down and pushing a shopping cart, follows closely behind Bob and accidentally bumps into his heel.
Image 3: [Camera 1] Shot from Alice's over-the-shoulder perspective. Alice is on the side closer to the camera, with only her shoulder appearing in the lower left corner of the frame. Bob is on the side farther from the camera, positioned slightly right of center in the frame. Bob quickly turns around, and his expression shifts from neutral to surprised.
Image 4: [Camera 2] Shot from Bob's over-the-shoulder perspective. Bob is on the side closer to the camera, with only his shoulder appearing in the lower right corner of the frame. Alice is on the side farther from the camera, positioned slightly left of center in the frame. Alice looks down, then up as she prepares to apologize. Upon realizing it's someone familiar, her expression shifts to one of surprise.
</SEQ_DESC>


[Output]
You need to select up to 8 of the most relevant reference images based on the user's description and put the corresponding indices in the ref_image_indices field of the output. At the same time, you should generate a text prompt that describes the image to be created, specifying which elements in the generated image should reference which image description (and which elements within it).

{format_instructions}


[Guidelines]
- Ensure that the language of all output values (not include keys) matches that used in the frame description.
- The reference image descriptions may depict the same character from different angles, in different outfits, or in different scenes. Identify the description closest to the version described by the user
- Prioritize image descriptions with similar compositions, i.e., shots taken by the same camera.
- The images from prior frames are arranged in chronological order. Give higher priority to more recent images (those closer to the end of the sequence).
- Choose reference image descriptions that are as concise as possible and avoid including duplicate information. For example, if Image 3 depicts the facial features of Bob from the front, and Image 1 also depicts Bob's facial features from the front-view portrait, then Image 1 is redundant and should not be selected.
- When a new character appears in the frame description, prioritize selecting their portrait image description (if available) to ensure accurate depiction of their appearance. Pay attention to whether the character is facing the camera from the front, side, or back. Choose the most suitable view as the reference image for the character.
- For character portraits, you can only select at most one image from multiple views (front, side, back). Choose the most appropriate one based on the frame description. For example, when depicting a character from the side, choose the side view of the character.
- Select at most **8** optimal reference image descriptions.
"""


system_prompt_template_select_reference_images_multimodal = \
"""
[Role]
You are a professional visual creation assistant skilled in multimodal image analysis and reasoning.

[Task]
Your core task is to intelligently select the most suitable reference images from a provided reference image library (including multiple character reference images and existing scene images from prior frames) based on the user's text description (describing the target frame), ensuring that the subsequently generated image meets the following key consistencies:
- Character Consistency: The appearance (e.g. gender, ethnicity, age, facial features, hairstyle, body shape), clothing, expression, posture, etc., of the generated character should highly match the reference images.
- Environmental Consistency: The scene of the generated image (e.g., background, lighting, atmosphere, layout) should remain coherent with the existing images from prior frames.
- Style Consistency: The visual style of the generated image (e.g., realistic, cartoon, film-like, color tone) should harmonize with the reference images and existing images.

[Input]
You will receive a text description of the target frame, along with a sequence of reference images.
- The text description of the target frame is enclosed within <FRAME_DESC> and </FRAME_DESC>.
- The sequence of reference images is enclosed within <SEQ_IMAGES> and </SEQ_IMAGES>. Each reference image is provided with a text description. The reference images are indexed starting from 0.

Below is an example of the input format:
<FRAME_DESC>
[Camera 1] Shot from Alice's over-the-shoulder perspective. <Alice> is on the side closer to the camera, with only her shoulder appearing in the lower left corner of the frame. <Bob> is on the side farther from the camera, positioned slightly right of center in the frame. <Bob>'s expression shifts from surprise to delight as he recognizes <Alice>.
</FRAME_DESC>

<SEQ_IMAGES>
Image 0: A front-view portrait of Alice.
[Image 0 here]
Image 1: A front-view portrait of Bob.
[Image 1 here]
Image 2: [Camera 0] Medium shot of the supermarket aisle. Alice and Bob are shown in profile facing the right side of the frame. Bob is on the right side of the frame, and Alice is on the left side. Alice, looking down and pushing a shopping cart, follows closely behind Bob and accidentally bumps into his heel.
[Image 2 here]
Image 3: [Camera 1] Shot from Alice's over-the-shoulder perspective. Alice is on the side closer to the camera, with only her shoulder appearing in the lower left corner of the frame. Bob is on the side farther from the camera, positioned slightly right of center in the frame. Bob is back to the camera.
[Image 3 here]
Image 4: [Camera 2] Shot from Bob's over-the-shoulder perspective. Bob is on the side closer to the camera, with only his shoulder appearing in the lower right corner of the frame. Alice is on the side farther from the camera, positioned slightly left of center in the frame. Alice looks down, then up as she prepares to apologize. Upon realizing it's someone familiar, her expression shifts to one of surprise.
</SEQ_IMAGES>

[Output]
You need to select the most relevant reference images based on the user's description and put the corresponding indices in the `ref_image_indices` field of the output. At the same time, you should generate a text prompt that describes the image to be created, specifying which elements in the generated image should reference which image (and which elements within it).

{format_instructions}


[Guidelines]
- Ensure that the language of all output values (not include keys) matches that used in the frame description.
- The reference image descriptions may depict the same character from different angles, in different outfits, or in different scenes. Identify the description closest to the version described by the user, select no more than one reference image for one character.
- Select no more than three images in total
- Prioritize image descriptions with similar compositions, i.e., shots taken by the same camera.
- The images from prior frames are arranged in chronological order. Give higher priority to more recent images (those closer to the end of the sequence).
- Choose reference image descriptions that are as concise as possible and avoid including duplicate information. For example, if Image 3 depicts the facial features of Bob from the front, and Image 1 also depicts Bob's facial features from the front-view portrait, then Image 1 is redundant and should not be selected.
- For character portraits, you can only select at most one image from multiple views (front, side, back). Choose the most appropriate one based on the frame description. For example, when depicting a character from the side, choose the side view of the character.
- Select at most **3** optimal reference image descriptions.
- The text guiding image editing should be as concise as possible.
- For text prompts involving reference image index, please use the new index after reference image selection instead of the old one.
- For text prompts, do not use any inappropriate or NSFW word.
"""


human_prompt_template_select_reference_images = \
"""
<FRAME_DESC>
{frame_description}
</FRAME_DESC>
"""




class RefImageIndicesAndTextPrompt(BaseModel):
    ref_image_indices: List[int] = Field(
        description="Indices of reference images selected from the provided images. For example, [0, 2, 5] means selecting the first, third, and sixth images. The indices should be 0-based.",
        examples=[
            [1, 3]
        ]
    )
    text_prompt: str = Field(
        description="Text description to guide the image generation. You need to describe the image to be generated, specifying which elements in the generated image should reference which image (and which elements within it). For example, 'Create an image following the given description: \nThe man is standing in the landscape. The man should reference Image 0. The landscape should reference Image 1.' Here, the index of the reference image should refer to its position in the ref_image_indices list, not the sequence number in the provided image list. Refer to the reference image must be in the format of Image N. Do not use any other word except Image.",
        examples=[
            "Create an image based on the following guidance: \n Make modifications based on Image 1: Bob's body turns to face the camera, while all other elements remain unchanged. Bob's appearance should refer to Image 0.",
            "Create an image following the given description: \nThe man is standing in the landscape. The man should reference Image 0. The landscape should reference Image 1."
        ]
    )

def _select_reference_images_and_generate_prompt(available_image_path_and_text_pairs: List[Tuple[str, str]],
                                               frame_description: str,):
    filtered_image_path_and_text_pairs = available_image_path_and_text_pairs
    # 1. filter images using text-only model
    if len(available_image_path_and_text_pairs) >= 8:
        human_content = []
        for idx, (_, text) in enumerate(available_image_path_and_text_pairs):
            human_content.append({
                "type": "text",
                "text": f"Image {idx}: {text}"
            })
        human_content.append({
            "type": "text",
            "text": human_prompt_template_select_reference_images.format(frame_description=frame_description)
        })
        parser = PydanticOutputParser(pydantic_object=RefImageIndicesAndTextPrompt)

        messages = [
            SystemMessage(content=system_prompt_template_select_reference_images_only_text.format(format_instructions=parser.get_format_instructions())),
            HumanMessage(content=human_content)
        ]

        chain = model | parser

        ref = chain.invoke(messages)
        filtered_image_path_and_text_pairs = [available_image_path_and_text_pairs[i] for i in ref.ref_image_indices]

    # 2. filter images using multimodal model
    human_content = []
    for idx, (image_path, text) in enumerate(filtered_image_path_and_text_pairs):
        human_content.append({
            "type": "text",
            "text": f"Image {idx}: {text}"
        })
        human_content.append({
            "type": "image_url",
            "image_url": {"url": encode_file(image_path)}
        })
    human_content.append({
        "type": "text",
        "text": human_prompt_template_select_reference_images.format(frame_description=frame_description)
    })

    parser = PydanticOutputParser(pydantic_object=RefImageIndicesAndTextPrompt)

    messages = [
        SystemMessage(content=system_prompt_template_select_reference_images_multimodal.format(format_instructions=parser.get_format_instructions())),
        HumanMessage(content=human_content)
    ]

    chain = model | parser

    response = chain.invoke(messages)       
    reference_image_path_and_text_pairs = [filtered_image_path_and_text_pairs[i] for i in response.ref_image_indices]
    return {
        "reference_image_path_and_text_pairs": reference_image_path_and_text_pairs,
        "text_prompt": response.text_prompt,
    }

def generate_frame_for_single_shot(image_output_path, selector_output_path, first_shot_ff_path_and_text_pair, frame_desc, visible_characters, character_portraits_registry):
    if os.path.exists(image_output_path):
        logging.info(f"🚀 Skipped generating frame, already exists.")
        pass
    else:
        available_image_path_and_text_pairs = []
        for visible_character in visible_characters:
            identifier_in_scene = visible_character.identifier_in_scene
            registry_item = character_portraits_registry[identifier_in_scene]
            for view, item in registry_item.items():
                available_image_path_and_text_pairs.append((item["path"], item["description"]))

        if first_shot_ff_path_and_text_pair is not None:
            available_image_path_and_text_pairs.append(first_shot_ff_path_and_text_pair)

        if os.path.exists(selector_output_path):
            with open(selector_output_path, 'r', encoding='utf-8') as f:
                selector_output = json.load(f)
            logging.info(f"🚀 Loaded existing reference image selection and prompt for frame from {selector_output_path}.")
        else:
            selector_output = _select_reference_images_and_generate_prompt(
                    available_image_path_and_text_pairs=available_image_path_and_text_pairs,
                    frame_description=frame_desc
                )
            with open(selector_output_path, 'w', encoding='utf-8') as f:
                json.dump(selector_output, f, ensure_ascii=False, indent=4)
            logging.info(f"☑️ Selected reference images and generated prompt for frame, saved to {selector_output_path}.")

    if not os.path.exists(image_output_path):
        reference_image_path_and_text_pairs, prompt = selector_output["reference_image_path_and_text_pairs"], selector_output["text_prompt"]
        prefix_prompt = ""
        for i, (image_path, text) in enumerate(reference_image_path_and_text_pairs):
            prefix_prompt += f"Image {i}: {text}\n"
        prompt = f"{prefix_prompt}\n{prompt}"
        reference_image_paths = [item[0] for item in reference_image_path_and_text_pairs]

        image2image(prompt, reference_image_paths, image_output_path)
        logging.info(f"☑️ Generated frame, saved to {image_output_path}.")

        
def select_reference_images_and_generate_prompt(state: VideoGenState) -> VideoGenState:
    for idx, script in enumerate(state["scene_desc"]):
        scene_root = os.path.join(state['cache_dir'], f"scene_{idx}")
        for j, camera in enumerate(state["camera_tree"][idx]):
            first_shot_idx = camera.active_shot_idxs[0]
            first_shot_ff_path = os.path.join(scene_root, f"shot_{first_shot_idx}", "first_frame.png")
            ff_selector_output_path = os.path.join(scene_root, f"shot_{first_shot_idx}", "first_frame_selector_output.json")
            logging.info(f"🖼️ Starting first_frame generation for shot {first_shot_idx}...")
            generate_frame_for_single_shot(first_shot_ff_path, ff_selector_output_path, None, state["shot_descriptions"][idx][first_shot_idx].ff_desc, [state["character_desc"][idx] for idx in state["shot_descriptions"][idx][first_shot_idx].ff_vis_char_idxs], state["character_images"])
            # if os.path.exists(first_shot_ff_path):
            #     pass
            # else:
            #     available_image_path_and_text_pairs = []

            #     for character_idx in state["shot_descriptions"][idx][first_shot_idx].ff_vis_char_idxs:
            #         identifier_in_scene = state["character_desc"][character_idx].identifier_in_scene
            #         registry_item = state["character_images"][identifier_in_scene]
            #         for view, item in registry_item.items():
            #             available_image_path_and_text_pairs.append((item["path"], item["description"]))
            #     if first_shot_ff_path_and_text_pair is not None:
            #         available_image_path_and_text_pairs.append(first_shot_ff_path_and_text_pair)

            #     ff_selector_output_path = os.path.join(scene_root, f"shot_{first_shot_idx}", "first_frame_selector_output.json")
            #     if os.path.exists(ff_selector_output_path):
            #         with open(ff_selector_output_path, 'r', encoding='utf-8') as f:
            #             ff_selector_output = json.load(f)
            #     else:
            #         ff_selector_output = _select_reference_images_and_generate_prompt(
            #             available_image_path_and_text_pairs=available_image_path_and_text_pairs,
            #             frame_description=state["shot_descriptions"][idx][first_shot_idx].ff_desc
            #         )
            #         with open(ff_selector_output_path, 'w', encoding='utf-8') as f:
            #             json.dump(ff_selector_output, f, ensure_ascii=False, indent=4)

            # if not os.path.exists(first_shot_ff_path):
            #     reference_image_path_and_text_pairs, prompt = ff_selector_output["reference_image_path_and_text_pairs"], ff_selector_output["text_prompt"]
            #     prefix_prompt = ""
            #     for i, (image_path, text) in enumerate(reference_image_path_and_text_pairs):
            #         prefix_prompt += f"Image {i}: {text}\n"
            #     prompt = f"{prefix_prompt}\n{prompt}"
            #     reference_image_paths = [item[0] for item in reference_image_path_and_text_pairs]

            #     image2image(prompt, reference_image_paths, first_shot_ff_path)

            if state["shot_descriptions"][idx][first_shot_idx].variation_type in ["medium", "large"]:
                last_shot_ff_path = os.path.join(scene_root, f"shot_{first_shot_idx}", "last_frame.png")
                lf_selector_output_path = os.path.join(scene_root, f"shot_{first_shot_idx}", "last_frame_selector_output.json")
                logging.info(f"🖼️ Starting last_frame generation for shot {first_shot_idx}...")
                generate_frame_for_single_shot(last_shot_ff_path, lf_selector_output_path, None, state["shot_descriptions"][idx][first_shot_idx].lf_desc, [state["character_desc"][idx] for idx in state["shot_descriptions"][idx][first_shot_idx].lf_vis_char_idxs], state["character_images"])

            for shot_idx in camera.active_shot_idxs[1:]:
                shot_path = os.path.join(scene_root, f"shot_{shot_idx}", "first_frame.png")
                ff_selector_output_path = os.path.join(scene_root, f"shot_{shot_idx}", "first_frame_selector_output.json")
                logging.info(f"🖼️ Starting first_frame generation for shot {shot_idx}...")
                generate_frame_for_single_shot(shot_path, ff_selector_output_path, (first_shot_ff_path, state["shot_descriptions"][idx][first_shot_idx].ff_desc), state["shot_descriptions"][idx][shot_idx].ff_desc, [state["character_desc"][idx] for idx in state["shot_descriptions"][idx][shot_idx].ff_vis_char_idxs], state["character_images"])
                if state["shot_descriptions"][idx][shot_idx].variation_type in ["medium", "large"]:
                    last_shot_path = os.path.join(scene_root, f"shot_{shot_idx}", "last_frame.png")
                    lf_selector_output_path = os.path.join(scene_root, f"shot_{shot_idx}", "last_frame_selector_output.json")
                    logging.info(f"🖼️ Starting last_frame generation for shot {shot_idx}...")
                    generate_frame_for_single_shot(last_shot_path, lf_selector_output_path, (shot_path, state["shot_descriptions"][idx][shot_idx].ff_desc), state["shot_descriptions"][idx][shot_idx].lf_desc, [state["character_desc"][idx] for idx in state["shot_descriptions"][idx][shot_idx].lf_vis_char_idxs], state["character_images"])
                    
                # if os.path.exists(shot_path):
                #     pass
                # else:
                #     available_image_path_and_text_pairs = []

                #     for character_idx in state["shot_descriptions"][idx][shot_idx].ff_vis_char_idxs:
                #         identifier_in_scene = state["character_desc"][character_idx].identifier_in_scene
                #         registry_item = state["character_images"][identifier_in_scene]
                #         for view, item in registry_item.items():
                #             available_image_path_and_text_pairs.append((item["path"], item["description"]))
    
                #     ff_selector_output_path = os.path.join(scene_root, f"shot_{shot_idx}", "first_frame_selector_output.json")
                #     if os.path.exists(ff_selector_output_path):
                #         with open(ff_selector_output_path, 'r', encoding='utf-8') as f:
                #             ff_selector_output = json.load(f)
                #     else:
                #         ff_selector_output = _select_reference_images_and_generate_prompt(
                #             available_image_path_and_text_pairs=available_image_path_and_text_pairs,
                #             frame_description=state["shot_descriptions"][idx][shot_idx].ff_desc
                #         )
                #         with open(ff_selector_output_path, 'w', encoding='utf-8') as f:
                #             json.dump(ff_selector_output, f, ensure_ascii=False, indent=4)
                # if not os.path.exists(shot_path):
                #     reference_image_path_and_text_pairs, prompt = ff_selector_output["reference_image_path_and_text_pairs"], ff_selector_output["text_prompt"]
                #     prefix_prompt = ""
                #     for i, (image_path, text) in enumerate(reference_image_path_and_text_pairs):
                #         prefix_prompt += f"Image {i}: {text}\n"
                #     prompt = f"{prefix_prompt}\n{prompt}"
                #     reference_image_paths = [item[0] for item in reference_image_path_and_text_pairs]
    
                #     image2image(prompt, reference_image_paths, shot_path)
                    

    return state