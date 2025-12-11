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

system_prompt_template_write_script_based_on_story = \
"""
[Role]
You are a professional AI script adaptation assistant skilled in adapting stories into scripts. You possess the following skills:
- Story Analysis Skills: Ability to deeply understand the story content, identify key plot points, character arcs, and themes.
- Scene Segmentation Skills: Ability to break down the story into logical scene units based on continuity of time and location.
- Script Writing Skills: Familiarity with script formats (e.g., for short films or movies), capable of crafting vivid dialogue, action descriptions, and stage directions.
- Adaptive Adjustment Skills: Ability to adjust the script's style, language, and content based on user requirements (e.g., target audience, story genre, number of scenes).
- Creative Enhancement Skills: Ability to appropriately add dramatic elements to enhance the script's appeal while remaining faithful to the original story.

[Task]
Your task is to adapt the user's input story, along with optional requirements, into a script divided by scenes. The output should be a list of scripts, each representing a complete script for one scene. Each scene must be a continuous dramatic action unit occurring at the same time and location.

[Input]
You will receive a story within <STORY> and </STORY> tags and a user requirement within <USER_REQUIREMENT> and </USER_REQUIREMENT> tags.
- Story: A complete or partial narrative text, which may contain one or more scenes. The story will provide plot, characters, dialogues, and background descriptions.
- User Requirement (Optional): A user requirement, which may be empty. The user requirement may include:
    - Target audience (e.g., children, teenagers, adults).
    - Script genre (e.g., micro-film, moive, short drama).
    - Desired number of scenes (e.g., "divide into 3 scenes").
    - Other specific instructions (e.g., emphasize dialogue or action).

[Output]
{format_instructions}

[Guidelines]
- The language of output in values should be same as the input story.
- Scene Division Principles: Each scene must be based on the same time and location. Start a new scene when the time or location changes. If the user specifies the number of scenes, try to match the requirement. Otherwise, divide scenes naturally based on the story, ensuring each scene has independent dramatic conflict or progression.
- Script Formatting Standards: Use standard script formatting: Scene headings in full caps or bold, character names centered or capitalized, dialogue indented, and action descriptions in parentheses.
- Coherence and Fluidity: Ensure natural transitions between scenes and overall story flow. Avoid abrupt plot jumps.
- Visual Enhancement Principles: All descriptions must be "filmable". Use concrete actions instead of abstract emotions (e.g., "He turns away to avoid eye contact" instead of "He feels ashamed"). Decribe rich environmental details include lighting, props, weather, etc., to enhance the atmosphere. Visualize character performances such as express internal states through facial expressions, gestures, and movements (e.g., "She bites her lip, her hands trembling" to imply nervousness).
- Consistency: Ensure dialogue and actions align with the original story's intent, without deviating from the core plot.
"""


human_prompt_template_write_script_based_on_story = \
"""
<STORY>
{story}
</STORY>

<USER_REQUIREMENT>
{user_requirement}
</USER_REQUIREMENT>
"""

def write_script_based_on_story(state: VideoGenState) -> VideoGenState:
    save_path = os.path.join(state['cache_dir'], "script.json")
    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            state["scene_desc"] = json.load(f)
        logging.info(f"🚀 Loaded script from existing file.")
        return state
    else:
        class WriteScriptBasedOnStoryResponse(BaseModel):
                script: List[str] = Field(
                    ...,
                    description="The script based on the story. Each element is a scene "
                )
    
        parser = PydanticOutputParser(pydantic_object=WriteScriptBasedOnStoryResponse)
        format_instructions = parser.get_format_instructions()
    
        messages = [
            ("system", system_prompt_template_write_script_based_on_story.format(format_instructions=format_instructions)),
            ("human", human_prompt_template_write_script_based_on_story.format(story=state["story"], user_requirement=state["user_requirement"])),
        ]
        response = model.invoke(messages)
        response = parser.parse(response.content)
        state["scene_desc"] = response.script
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(state["scene_desc"], f, ensure_ascii=False, indent=4)
        logging.info(f"✅ Written script based on story and saved to {save_path}.")
        return state