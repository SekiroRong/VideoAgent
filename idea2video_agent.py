from typing import Literal
from collections import defaultdict
from langgraph.graph import StateGraph, START, END
# from agents import story_writer, character_extractor, character_portraits_generator, scene_writer, scene2video
from agents import develop_story, extract_characters, generate_character_images, write_script_based_on_story, design_storyboard, design_shot, construct_camera_tree, VideoGenState
from agents import select_reference_images_and_generate_prompt, generate_single_video, merge_final_video
from langchain.messages import AnyMessage
import operator
import os
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

def approval_node(state: VideoGenState):
    # Pause execution; payload shows up under result["__interrupt__"]
    (is_approved, reason) = interrupt("Do you want to proceed with this action?")

    # Route based on the response
    if is_approved:
        return True  # Runs after the resume payload is provided
    else:
        state["need_regen"]["develop_story"] = [True, reason]
        return False


# Build workflow
agent_builder = StateGraph(VideoGenState)

# Add nodes
agent_builder.add_node("develop_story", develop_story)
agent_builder.add_node("extract_characters", extract_characters)
agent_builder.add_node("generate_character_images", generate_character_images)
agent_builder.add_node("write_script_based_on_story", write_script_based_on_story)
agent_builder.add_node("design_storyboard", design_storyboard)
agent_builder.add_node("design_shot", design_shot)
agent_builder.add_node("construct_camera_tree", construct_camera_tree)
agent_builder.add_node("select_reference_images_and_generate_prompt", select_reference_images_and_generate_prompt)
agent_builder.add_node("generate_single_video", generate_single_video)
agent_builder.add_node("merge_final_video", merge_final_video)
agent_builder.add_node("approval_node", approval_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "develop_story")
# agent_builder.add_edge("develop_story", "approval_node")
agent_builder.add_conditional_edges("develop_story", approval_node, {True: "extract_characters", False: "develop_story"})
# agent_builder.add_edge("develop_story", "extract_characters")
agent_builder.add_edge("extract_characters", "generate_character_images")
agent_builder.add_edge("generate_character_images", "write_script_based_on_story")
agent_builder.add_edge("write_script_based_on_story", "design_storyboard")
agent_builder.add_edge("design_storyboard", "design_shot")
agent_builder.add_edge("design_shot", "construct_camera_tree")
agent_builder.add_edge("construct_camera_tree", "select_reference_images_and_generate_prompt")
agent_builder.add_edge("select_reference_images_and_generate_prompt", "generate_single_video")
agent_builder.add_edge("generate_single_video", "merge_final_video")
agent_builder.add_edge("merge_final_video", END)

# Compile the agent
checkpointer = MemorySaver()
agent = agent_builder.compile(checkpointer=checkpointer)

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

config = {"configurable": {"thread_id": "approval-123"}}
resumed = agent.invoke({"user_idea": user_idea, "user_requirement": user_requirement, "style": style,"cache_dir": cache_dir, "need_regen": defaultdict(lambda: [False, ""])}, config=config,)
# 命令行接收用户输入（处理输入合法性）
while "__interrupt__" in resumed.keys():
    user_input = input("Do you want to proceed with this action? (y/n)：").strip().lower()
    if user_input in ["y", "n"]:
        is_approved = (user_input == "y")
        if is_approved:
            resumed = agent.invoke(Command(resume=(is_approved, "")), config=config)
        else:
            user_input = input("Why?：").strip().lower()
            resumed = agent.invoke(Command(resume=(is_approved, user_input)), config=config)
    else:
        print("Invalid Input， y or n !")
# messages = agent.invoke({"user_idea": [HumanMessage(content=user_idea)], "user_requirement": [HumanMessage(content=user_requirement)], "style": [HumanMessage(content=style)],"cache_dir": cache_dir})
# print(messages)