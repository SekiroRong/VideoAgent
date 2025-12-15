from typing import TypedDict, List, Dict, Any, DefaultDict

# 定义状态数据结构 - 贯穿整个工作流的核心数据
class VideoGenState(TypedDict):
    """视频生成Agent的状态定义"""
    user_idea: str                # 用户初始创意
    user_requirement: str                # 用户初始创意
    style: str
    story: str                    # 生成的故事文本
    character_desc: List[Dict[str, Any]] # 人物描述（主要/次要）
    character_images: Dict[str, Any]   # 生成的人物图像URL/路径
    story_board: List[Any]
    shot_descriptions: List[Any]
    camera_tree: List[Any]
    scene_desc: List[str] # 场景描述列表
    final_video: str              # 最终视频URL/路径
    cache_dir: str
    need_regen: DefaultDict[str, bool]