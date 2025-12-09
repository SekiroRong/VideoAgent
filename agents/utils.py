import json
import os
import dashscope
from http import HTTPStatus
from dashscope import MultiModalConversation
from dashscope import VideoSynthesis
import mimetypes
import requests

# 以下为北京地域url，若使用新加坡地域的模型，需将url替换为：https://dashscope-intl.aliyuncs.com/api/v1
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

# 新加坡和北京地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
# 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
api_key = os.getenv("DASHSCOPE_API_KEY")

def text2image(prompt, save_dir):
    messages = [
        {
            "role": "user",
            "content": [
                {"text": prompt}
            ]
        }
    ]
    
    response = MultiModalConversation.call(
        api_key=api_key,
        model="qwen-image-plus",
        messages=messages,
        result_format='message',
        stream=False,
        watermark=False,
        prompt_extend=True,
        negative_prompt='',
        size='1328*1328'
    )
    
    if response.status_code == 200:
        response = json.dumps(response, ensure_ascii=False)
        response_dict = json.loads(response)
        print(response_dict)
        image_url = response_dict["output"]["choices"][0]["message"]["content"][0]["image"]
        print(image_url)
        # 注意：添加超时和请求头，避免下载失败
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        img_response = requests.get(
            image_url,
            headers=headers,
            timeout=30,  # 超时时间30秒
            stream=True  # 流式下载（适合大文件）
        )
        img_response.raise_for_status()  # 抛出HTTP错误（如404/500）

        # 写入文件（二进制模式）
        with open(save_dir, "wb") as f:
            for chunk in img_response.iter_content(chunk_size=8192):  # 分块写入
                f.write(chunk)

        print(f"图片已保存到: {save_dir}")
    else:
        print(f"HTTP返回码：{response.status_code}")
        print(f"错误码：{response.code}")
        print(f"错误信息：{response.message}")
        print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")

# --- 辅助函数：用于 Base64 编码 ---
# 格式为 data:{MIME_type};base64,{base64_data}
def encode_file(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError("不支持或无法识别的图像格式")
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{encoded_string}"

"""
图像输入方式说明：
以下提供了三种图片输入方式，三选一即可

1. 使用公网URL - 适合已有公开可访问的图片
2. 使用本地文件 - 适合本地开发测试
3. 使用Base64编码 - 适合私有图片或需要加密传输的场景
"""

# 【方式一】使用公网可访问的图片URL
# 示例：使用一个公开的图片URL
img_url = "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250925/wpimhv/rap.png"

# 【方式二】使用本地文件（支持绝对路径和相对路径）
# 格式要求：file:// + 文件路径
# 示例（绝对路径）：
# img_url = "file://" + "/path/to/your/img.png"    # Linux/macOS
# img_url = "file://" + "C:/path/to/your/img.png"  # Windows
# 示例（相对路径）：
# img_url = "file://" + "./img.png"                # 相对当前执行文件的路径

# 【方式三】使用Base64编码的图片
# img_url = encode_file("./img.png")

# 设置音频audio url
audio_url = "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250925/ozwpvi/rap.mp3"

def sample_call_i2v():
    # 同步调用，直接返回结果
    print('please wait...')
    rsp = VideoSynthesis.call(api_key=api_key,
                              model='wan2.5-i2v-preview',
                              prompt='一幅都市奇幻艺术的场景。一个充满动感的涂鸦艺术角色。一个由喷漆所画成的少年，正从一面混凝土墙上活过来。他一边用极快的语速演唱一首英文rap，一边摆着一个经典的、充满活力的说唱歌手姿势。场景设定在夜晚一个充满都市感的铁路桥下。灯光来自一盏孤零零的街灯，营造出电影般的氛围，充满高能量和惊人的细节。视频的音频部分完全由他的rap构成，没有其他对话或杂音。',
                              img_url=img_url,
                              audio_url=audio_url,
                              resolution="480P",
                              duration=10,
                              # audio=True, # 可选，因为模型默认开启自动配音
                              prompt_extend=True,
                              watermark=False,
                              negative_prompt="",
                              seed=12345)
    print(rsp)
    if rsp.status_code == HTTPStatus.OK:
        print("video_url:", rsp.output.video_url)
    else:
        print('Failed, status_code: %s, code: %s, message: %s' %
              (rsp.status_code, rsp.code, rsp.message))

if __name__ == "__main__":
    text2image("爱因斯坦拳打皮卡丘", "output.png")