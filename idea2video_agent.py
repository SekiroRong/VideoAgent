from typing import Literal
from langgraph.graph import StateGraph, START, END
# from agents import story_writer, character_extractor, character_portraits_generator, scene_writer, scene2video
from agents import develop_story, extract_characters, generate_character_images, write_script_based_on_story, design_storyboard, VideoGenState
from langchain.messages import AnyMessage
import operator
import os



# Build workflow
agent_builder = StateGraph(VideoGenState)

# Add nodes
agent_builder.add_node("develop_story", develop_story)
agent_builder.add_node("extract_characters", extract_characters)
agent_builder.add_node("generate_character_images", generate_character_images)
agent_builder.add_node("write_script_based_on_story", write_script_based_on_story)
agent_builder.add_node("design_storyboard", design_storyboard)

# Add edges to connect nodes
agent_builder.add_edge(START, "develop_story")
agent_builder.add_edge("develop_story", "extract_characters")
agent_builder.add_edge("extract_characters", "generate_character_images")
agent_builder.add_edge("generate_character_images", "write_script_based_on_story")
agent_builder.add_edge("write_script_based_on_story", "design_storyboard")
agent_builder.add_edge("design_storyboard", END)

# Compile the agent
agent = agent_builder.compile()

user_idea = \
    """
A beaufitul fit woman with black hair, great butt and thigs is exercising in a
gym surrounded by glass windows with a beautiful beach view on the outside.
She is performing glute exercises that highlight her beautiful back and sexy outfit
and showing the audience the proper form. Between the 3 different exercises she looks
at the camera with a gorgeous look asking the viewer understood the proper form.
"""
user_requirement = \
    """
For adults, do not exceed 3 scenes. Each scene should be no more than 5 shots.
"""
style = "Realistic, warm feel"

cache_dir = "working_dir"
os.makedirs(cache_dir, exist_ok=True)
from langchain.messages import HumanMessage
messages = agent.invoke({"user_idea": user_idea, "user_requirement": user_requirement, "style": style,"cache_dir": cache_dir})
# messages = agent.invoke({"user_idea": [HumanMessage(content=user_idea)], "user_requirement": [HumanMessage(content=user_requirement)], "style": [HumanMessage(content=style)],"cache_dir": cache_dir})
# print(messages)