# VideoAgent
### 🌟 Overview

This repository is a complete refactoring of [ViMax](https://github.com/HKUDS/ViMax/) using LangGraph, enabling more robust, modular, and scalable agentic video generation workflows.

ViMax transforms your creative concepts into complete videos through an intelligent multi-agent system, handling scriptwriting, storyboarding, character design, and video generation—all end-to-end with enhanced LangGraph-powered agent coordination.

### ✨ Key Features

- LangGraph-Powered Agents: More flexible and maintainable agent workflows with clear state management
- End-to-End Automation: From text prompt to finished video with minimal human intervention
- Enhanced Consistency: Improved character and scene consistency across multi-shot videos
- Modular Design: Easy to extend or modify individual agent components
- Parallel Processing: Efficient video generation pipeline with parallelizable tasks

### 🎯 Usage

How to acquire API:[bailian](https://bailian.console.aliyun.com/?spm=5176.29597918.0.0.65a27b080z86SK&tab=api#/api)

```bash
export DASHSCOPE_API_KEY='your api key'
export DASHSCOPE_API_BASE='your api base'
```

### 🎬 Outputs

![output.gif](assets/final_video.gif)

```bash
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
```
