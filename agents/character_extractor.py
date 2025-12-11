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

system_prompt_template_extract_characters = \
"""
[Role]
You are a top-tier movie script analysis expert.

[Task]
Your task is to analyze the provided script and extract all relevant character information.

[Input]
You will receive a script enclosed within <SCRIPT> and </SCRIPT>.

Below is a simple example of the input:

<SCRIPT>
A young woman sits alone at a table, staring out the window. She takes a sip of her coffee and sighs. The liquid is no longer warm, just a bitter reminder of the time that has passed. Outside, the world moves in a blur of hurried footsteps and distant car horns, but inside the quiet café, time feels thick and heavy.
Her finger traces the rim of the ceramic mug, following the imperfect circle over and over. The decision she had to make was supposed to be simple—a mere checkbox on the form of her life. Yesor No. Stayor Go. Yet, it had rooted itself in her chest, a tangled knot of fear and longing.
</SCRIPT>

[Output]
{format_instructions}


[Guidelines]
- Ensure that the language of all output values(not include keys) matches that used in the script.
- Group all names referring to the same entity under one character. Select the most appropriate name as the character's identifier. If the person is a real famous person, the real person's name should be retained (e.g., Elon Musk, Bill Gates)
- If the character's name is not mentioned, you can use reasonable pronouns to refer to them, including using their occupation or notable physical traits. For example, "the young woman" or "the barista".
- For background characters in the script, you do not need to consider them as individual characters.
- If a character's traits are not described or only partially outlined in the script, you need to design plausible features based on the context to make their characteristics more complete and detailed, ensuring they are vivid and evocative.
- In static features, you need to describe the character's physical appearance, physique, and other relatively unchanging features. In dynamic features, you need to describe the character's attire, accessories, key items they carry, and other easily changeable features.
- Don't include any information about the character's personality, role, or relationships with others in either static or dynamic features.
- When designing character features, within reasonable limits, different character appearances should be made more distinct from each other.
- The description of characters should be detailed, avoiding the use of abstract terms. Instead, employ descriptions that can be visualized—such as specific clothing colors and concrete physical traits (e.g., large eyes, a high nose bridge).
- The description of characters must include all attributes required, including idx, identifier_in_scene, is_visible, static_features, dynamic_features, and those attributes must be valid
"""

human_prompt_template_extract_characters = \
"""
<SCRIPT>
{script}
</SCRIPT>
"""

class CharacterInScene(BaseModel):
    idx: int = Field(
        description="The index of the character in the scene, starting from 0",
    )
    identifier_in_scene: str = Field(
        description="The identifier for the character in this specific scene, which may differ from the base identifier",
        examples=["Alice", "Bob the Builder"],
    )
    is_visible: bool = Field(
        description="Indicates whether the character is visible in this scene",
        examples=[True, False],
    )
    static_features: str = Field(
        description="The static features of the character in this specific scene, such as facial features and body shape that remain constant or are rarely changed. If the character is not visible, this field can be left empty.",
        examples=[
            "Alice has long blonde hair and blue eyes, and is of slender build.",
            "Bob the Builder is a middle-aged man with a sturdy build.",
        ]
    )
    dynamic_features: Optional[str] = Field(
        description="The dynamic features of the character in this specific scene, such as clothing and accessories that may change from scene to scene. If not mentioned, this field can be left empty. If the character is not visible, this field should be None.",
        examples=[
            "Wearing a red scarf and a black leather jacket",
        ]
    )

    def __str__(self):
        # Alice[visible]
        # static features: Alice has long blonde hair and blue eyes, and is of slender build.
        # dynamic features: Wearing a red scarf and a black leather jacket

        s = f"{self.identifier_in_scene}"
        s += "[visible]" if self.is_visible else "[not visible]"
        s += "\n"
        s += f"static features: {self.static_features}\n"
        s += f"dynamic features: {self.dynamic_features}\n"

        return s

class ExtractCharactersResponse(BaseModel):
    characters: List[CharacterInScene] = Field(
        ..., description="A list of characters extracted from the script."
    )

def extract_characters(state: VideoGenState) -> VideoGenState:
    save_path = os.path.join(state['cache_dir'], "characters.json")
    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            characters = json.load(f)
        characters = [CharacterInScene.model_validate(
                character) for character in characters]
        state["character_desc"] = characters
        logging.info(f"🚀 Loaded {len(characters)} characters from existing file.")
    else:
        parser = PydanticOutputParser(pydantic_object=ExtractCharactersResponse)
        
        messages = [
            SystemMessage(content=system_prompt_template_extract_characters.format(format_instructions=parser.get_format_instructions())),
            HumanMessage(content=human_prompt_template_extract_characters.format(script=state["story"])),
        ]
    
        chain = model | parser
    
        response: ExtractCharactersResponse = chain.invoke(messages)
    
        characters = response.characters
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump([character.model_dump()
                      for character in characters], f, ensure_ascii=False, indent=4)
        state["character_desc"] = characters
        logging.info(f"✅ Extracted {len(characters)} characters from story and saved to {save_path}.")
    return state