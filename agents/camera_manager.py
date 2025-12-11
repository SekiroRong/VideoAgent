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

model = ChatQwen(
    model="qwen-flash",
    max_tokens=3_000,
    timeout=None,
    max_retries=2,
    # other params...
)

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict
import json

from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Tuple

from .interfaces import Camera

system_prompt_template_select_reference_camera = \
"""
[Role]
You are a professional video editing expert specializing in multi-camera shot analysis and scene structure modeling. You have deep knowledge of cinematic language, enabling you to understand shot sizes (e.g., wide shot, medium shot, close-up) and content inclusion relationships. You can infer hierarchical structures between camera positions based on corresponding shot descriptions.

[Task]
Your task is to analyze the input camera position data to construct a "camera position tree". This tree structure represents a relationship where a parent camera's content encompasses that of a child camera. Specifically, you need to identify the parent camera for each camera position (if one exists) and determine the dependent shot indices (i.e., the specific shots within the parent camera's footage that contain the child camera's content). If a camera position has no parent, output None.

[Input]
The input is a sequence of cameras. The sequence will be enclosed within <CAMERA_SEQ> and </CAMERA_SEQ>.
Each camera contains a sequence of shots filmed by the camera, which will be enclosed within <CAMERA_N> and </CAMERA_N>, where N is the index of the camera.

Below is an example of the input format:

<CAMERA_SEQ>
<CAMERA_0>
Shot 0: Medium shot of the street. Alice and Bob are walking towards each other.
Shot 2: Medium shot of the street. Alice and Bob hug each other.
</CAMERA_0>
<CAMERA_1>
Shot 1: Close-up of the Alice's face. Her expression shifts from surprise to delight as she recognizes Bob.
</CAMERA_1>
</CAMERA_SEQ>


[Output]
{format_instructions}

[Guidelines]
- The language of all output values (not include keys) should be consistent with the language of the input.
- Content Inclusion Check: The parent camera should as fully as possible contain the child camera's content in certain shots (e.g., a parent medium two-shot encompasses a child over-the-shoulder reverse shot). Analyze shot descriptions by comparing keywords (e.g., characters, actions, setting) to ensure the parent shot's field of view covers the child shot's.
- Transition Smoothness Priority: Larger shot size as parent camera is preferred, such as Wide Shot -> Medium Shot or Medium Shot -> Close-up. The shot sizes of adjacent parent and child nodes should be as similar as possible. A direct transition from a long shot to a close-up is not allowed unless absolutely necessary.
- Temporal Proximity: Each camera is described by its corresponding first shot, and the parent camera is located based on the description of the first shot. The shot index of the parent camera should be as close as possible to the first shot index of the child camera.
- Logical Consistency: The camera tree should be acyclic, avoid circular dependencies. If a camera is contained by multiple potential parents, select the best match (based on shot size and content). If there is no suitable parent camera, output None.
- When a broader perspective is not available, choose the shot with the largest overlapping field of view as the parent (the one with the most information overlap), or a shot can also serve as the parent of a reverse shot. When two cameras can be the parent of each other, choose the one with the smaller index as the parent of the camera with the larger index.
- Only one camera can exist without a parent.
- When describing the elements lost in a shot, carefully compare the details between the parent shot and the child shot. For example, the parent shot is a medium shot of Character A and Character B facing each other (both in profile to the camera), while the child shot is a close-up of Character A (with Character A facing the camera directly). In this case, the child shot lacks the frontal view information of Character A.
- The first camera must be the root of the camera tree.
"""


human_prompt_template_select_reference_camera = \
"""
<CAMERA_SEQ>
{camera_seq_str}
</CAMERA_SEQ>
"""


class CameraParentItem(BaseModel):
    parent_cam_idx: Optional[int] = Field( 
        default=None, 
        description="The index of the parent camera. Set to None if the camera has no parent (e.g., for a root camera).",
        examples=[0, 1, None], 
    )
    parent_shot_idx: Optional[int] = Field( 
        default=None, 
        description="The index of the dependent shot. Set to None if the camera has no parent (e.g., for a root camera).",
        examples=[0, 3, None], 
    )
    reason: str = Field(
        description="The reason for the selection of the parent camera. If the camera has no parent, it should explain why it's a root camera.",
        examples=[
            "The parent shot's field of view covers the child shot's field of view (from medium shot to close-up)",
            "The parent shot and the child shot have a shot/reverse shot relationship.",
            "CAMERA_0 (Shot 0) establishes the entire scene and contains all characters and the setting. It is the root camera." # 补充 LLM 实际输出的例子
        ],
    )
    is_parent_fully_covers_child: Optional[bool] = Field( 
        default=None, 
        description="Whether the parent camera fully covers the child camera's content. Set to None if the camera has no parent.",
        examples=[True, False, None], 
    )
    missing_info: Optional[str] = Field(
        default=None,
        description="The missing elements in the child shot that are not covered by the parent shot. If the parent shot fully covers the child shot, set this to None.",
        examples=[
            "The frontal view of Alice.",
            None,
        ],
    )

class CameraTreeResponse(BaseModel):
    camera_parent_items: List[Optional[CameraParentItem]] = Field(
        description="The parent camera items for each camera. If a camera has no parent, set this to None. The length of the list should be the same as the number of cameras.",
    )

def construct_camera_tree(state: VideoGenState) -> VideoGenState:
    parser = PydanticOutputParser(pydantic_object=CameraTreeResponse)

    state["camera_tree"] = []
    for idx, shot_descriptions in enumerate(state["shot_descriptions"]):
        save_root = os.path.join(state['cache_dir'], f"scene_{idx}")
        save_path = os.path.join(save_root, "camera_tree.json")
        if os.path.exists(save_path):
            with open(save_path, 'r', encoding='utf-8') as f:
                camera_tree = json.load(f)
            camera_tree = [Camera.model_validate(camera) for camera in camera_tree]
            state["camera_tree"].append(camera_tree)
            logging.info(f"🚀 Loaded {len(camera_tree)} cameras from existing file.")
        else:
            cameras: List[Camera] = []
            
            for shot_description in shot_descriptions:
                if shot_description.cam_idx not in [camera.idx for camera in cameras]:
                    cameras.append(Camera(idx=shot_description.cam_idx, active_shot_idxs=[shot_description.idx]))
                else:
                    cameras[shot_description.cam_idx].active_shot_idxs.append(shot_description.idx)

            camera_seq_str = "<CAMERA_SEQ>\n"
            for cam in cameras:
                camera_seq_str += f"<CAMERA_{cam.idx}>\n"
                for shot_idx in cam.active_shot_idxs:
                    camera_seq_str += f"Shot {shot_idx}: {shot_descriptions[shot_idx].visual_desc}\n"
                camera_seq_str += f"</CAMERA_{cam.idx}>\n"
            camera_seq_str += "</CAMERA_SEQ>"

            messages = [
                SystemMessage(content=system_prompt_template_select_reference_camera.format(format_instructions=parser.get_format_instructions())),
                HumanMessage(content=human_prompt_template_select_reference_camera.format(camera_seq_str=camera_seq_str)),
            ]
    
            chain = model | parser
            response: CameraTreeResponse = chain.invoke(messages)
            for cam, parent_cam_item in zip(cameras, response.camera_parent_items):
                cam.parent_cam_idx = parent_cam_item.parent_cam_idx if parent_cam_item is not None else None
                cam.parent_shot_idx = parent_cam_item.parent_shot_idx if parent_cam_item is not None else None
                cam.reason = parent_cam_item.reason if parent_cam_item is not None else None
                cam.parent_shot_idx = parent_cam_item.parent_shot_idx if parent_cam_item is not None else None
                cam.is_parent_fully_covers_child = parent_cam_item.is_parent_fully_covers_child if parent_cam_item is not None else None
                cam.missing_info = parent_cam_item.missing_info if parent_cam_item is not None else None

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump([camera.model_dump() for camera in cameras], f, ensure_ascii=False, indent=4)

            state["camera_tree"].append(cameras)
            logging.info(f"✅ Constructed camera tree and saved to {save_path}.")

    return state