import getpass
import os

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

from .interfaces import ShotBriefDescription

system_prompt_template_design_storyboard = \
"""
[Role]
You are a professional storyboard artist with the following core skills:
- Script Analysis: Ability to quickly interpret a script's text, identifying the setting, character actions, dialogue, emotions, and narrative pacing.
- Visualization: Expertise in translating written descriptions into visual frames, including composition, lighting, and spatial arrangement.
- Storyboarding: Proficiency in cinematic language, such as shot types (e.g., close-up, medium shot, wide shot), camera angles (e.g., high angle, eye-level), camera movements (e.g., zoom, pan), and transitions.
- Narrative Continuity: Ability to ensure the storyboard sequence is logically smooth, highlights key plot points, and maintains emotional consistency.
- Technical Knowledge: Understanding of basic storyboard formats and industry standards, such as using numbered shots and concise descriptions.

[Task]
Your task is to design a complete storyboard based on a user-provided script (which contains only one scene). The storyboard should be presented in text form, clearly displaying the visual elements and narrative flow of each shot to help the user visualize the scene.

[Input]
The user will provide the following input.
- Script:A complete scene script containing dialogue, action descriptions, and scene settings. The script focuses on only one scene; there is no need to handle multiple scene transitions. The script input is enclosed within <SCRIPT> and </SCRIPT>.
- Characters List: A list describing basic information for each character, such as name, personality traits, appearance (if relevant). The character list is enclosed within <CHARACTERS> and </CHARACTERS>.
- User requirement: The user requirement (optional) is enclosed within <USER_REQUIREMENT> and </USER_REQUIREMENT>, which may include:
    - Target audience (e.g., children, teenagers, adults).
    - Storyboard style (e.g., realistic, cartoon, abstract).
    - Desired number of shots (e.g., "not more than 10 shots").
    - Other specific instructions (e.g., emphasize the characters' actions).

[Output]
{format_instructions}

[Guidelines]
- Ensure all output values (except keys) match the language used in the script.
- Each shot must have a clear narrative purpose—such as establishing the setting, showing character relationships, or highlighting reactions.
- Use cinematic language deliberately: close-ups for emotion, wide shots for context, and varied angles to direct audience attention.
- When designing a new shot, first consider whether it can be filmed using an existing camera position. Introduce a new one only if the shot size, angle, and focus differ significantly. If the camera undergoes significant movement, it cannot be used thereafter.
- Keep character names in visual descriptions and speaker fields consistent with the character list. In visual descriptions, enclose names in angle brackets (e.g., <Alice>), but not in dialogue or speaker fields.
- When describing visual elements, it is necessary to indicate the position of the element within the frame. For example, Character A is on the left side of the frame, facing toward the right, with a table in front of him. The table is positioned slightly to the left of the center of the frame. Ensure that invisible elements are not included. For instance, do not describe someone behind a closed door if they cannot be seen.
- Avoid unsafe content (violence, discrimination, etc.) in visual descriptions. Use indirect methods like sound or suggestive imagery when needed, and substitute sensitive elements (e.g., ketchup for blood).
- Assign at most one dialogue line per character per shot. Each line of dialogue should correspond to a shot.
- Each shot requires an independent description without reference to each other.
- When the shot focuses on a character, describe which specific body part the focus is on.
- When describing a character, it is necessary to indicate the direction they are facing.
"""


human_prompt_template_design_storyboard = \
"""
<SCRIPT>
{script_str}
</SCRIPT>

<CHARACTERS>
{characters_str}
</CHARACTERS>

<USER_REQUIREMENT>
{user_requirement_str}
</USER_REQUIREMENT>
"""

def design_storyboard(state: VideoGenState) -> VideoGenState:
    class StoryboardResponse(BaseModel):
            storyboard: List[ShotBriefDescription] = Field(
                description="A complete storyboard of the scene, including the visual and audio description of each shot.",
            )
    state["story_board"] = []
    for idx, script in enumerate(state["scene_desc"]):
        save_root = os.path.join(state['cache_dir'], f"scene_{idx}")
        os.makedirs(save_root, exist_ok=True)
        save_path = os.path.join(save_root, "storyboard.json")
        if os.path.exists(save_path):
            with open(save_path, 'r', encoding='utf-8') as f:
                storyboard = json.load(f)
            storyboard = [ShotBriefDescription.model_validate(shot) for shot in storyboard]
            state["story_board"].append(storyboard)
        else:
            script_str = script.strip()
            characters_str = "\n".join([f"Character {index}: {char}" for index, char in enumerate(state["character_desc"])])
            user_requirement_str = state["user_requirement"].strip() if state["user_requirement"] else ""
    
            parser = PydanticOutputParser(pydantic_object=StoryboardResponse)
            messages = [
                ('system', system_prompt_template_design_storyboard.format(format_instructions=parser.get_format_instructions())),
                ('human', human_prompt_template_design_storyboard.format(script_str=script_str, characters_str=characters_str, user_requirement_str=user_requirement_str)),
            ]
            chain = model | parser
            
            response: StoryboardResponse = chain.invoke(messages)
    
            storyboard = response.storyboard
    
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump([shot.model_dump() for shot in storyboard], f, ensure_ascii=False, indent=4)
            state["story_board"].append(storyboard)

    return state