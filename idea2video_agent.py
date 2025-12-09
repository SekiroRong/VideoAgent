from typing import Literal
from langgraph.graph import StateGraph, START, END
from agents import story_writer, character_extractor, character_portraits_generator, scene_writer, scene2video
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator


# 定义状态数据结构 - 贯穿整个工作流的核心数据
class VideoGenState(TypedDict):
    """视频生成Agent的状态定义"""
    user_idea: str                # 用户初始创意
    user_requirement: str                # 用户初始创意
    story: str                    # 生成的故事文本
    character_desc: Dict[str, Any] # 人物描述（主要/次要）
    character_images: List[str]   # 生成的人物图像URL/路径
    scene_desc: List[Dict[str, Any]] # 场景描述列表
    final_video: str              # 最终视频URL/路径


# Build workflow
agent_builder = StateGraph(VideoGenState)

# Add nodes
agent_builder.add_node("story_writer", story_writer)
agent_builder.add_node("character_extractor", character_extractor)
agent_builder.add_node("character_portraits_generator", character_portraits_generator)
agent_builder.add_node("scene_writer", scene_writer)
agent_builder.add_node("scene2video", scene2video)

# Add edges to connect nodes
agent_builder.add_edge(START, "story_writer")
agent_builder.add_edge("story_writer", "character_extractor")
agent_builder.add_edge("character_extractor", "character_portraits_generator")
agent_builder.add_edge("character_portraits_generator", "scene_writer")
agent_builder.add_edge("scene_writer", "scene2video")
agent_builder.add_edge("scene2video", END)

# Compile the agent
agent = agent_builder.compile()