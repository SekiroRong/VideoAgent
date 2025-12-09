from typing import Literal
from langgraph.graph import StateGraph, START, END
# from agents import story_writer, character_extractor, character_portraits_generator, scene_writer, scene2video
from agents import develop_story, VideoGenState
from langchain.messages import AnyMessage
import operator



# Build workflow
agent_builder = StateGraph(VideoGenState)

# Add nodes
agent_builder.add_node("develop_story", develop_story)
# agent_builder.add_node("character_extractor", character_extractor)
# agent_builder.add_node("character_portraits_generator", character_portraits_generator)
# agent_builder.add_node("scene_writer", scene_writer)
# agent_builder.add_node("scene2video", scene2video)

# Add edges to connect nodes
agent_builder.add_edge(START, "develop_story")
# agent_builder.add_edge("story_writer", "character_extractor")
# agent_builder.add_edge("character_extractor", "character_portraits_generator")
# agent_builder.add_edge("character_portraits_generator", "scene_writer")
# agent_builder.add_edge("scene_writer", "scene2video")
agent_builder.add_edge("develop_story", END)

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

from langchain.messages import HumanMessage
messages = agent.invoke({"user_idea": [HumanMessage(content=user_idea)], "user_requirement": [HumanMessage(content=user_requirement)]})
print(messages)