import json
import os
import dashscope
from http import HTTPStatus
from dashscope import MultiModalConversation
from dashscope import VideoSynthesis
import mimetypes
import base64
import requests

# 以下为北京地域url，若使用新加坡地域的模型，需将url替换为：https://dashscope-intl.aliyuncs.com/api/v1
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

# ---用于 Base64 编码 ---
# 格式为 data:{mime_type};base64,{base64_data}
def encode_file(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError("不支持或无法识别的图像格式")

    try:
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(
                image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"
    except IOError as e:
        raise IOError(f"读取文件时出错: {file_path}, 错误: {str(e)}")

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

def image2image(prompt, image_paths, save_dir):
    content = [{"image": encode_file(image_path)} for image_path in image_paths]
    content.append({"text": prompt})
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    
    # qwen-image-edit-plus支持输出1-6张图片，此处以2张为例
    response = MultiModalConversation.call(
        api_key=api_key,
        model="qwen-image-edit-plus",
        messages=messages,
        stream=False,
        n=1,
        watermark=False,
        negative_prompt=" ",
        prompt_extend=True,
        # 仅当输出图像数量n=1时支持设置size参数，否则会报错
        # size="2048*1024",
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


def sample_call_i2v(prompt, image_paths, save_dir):
    # 同步调用，直接返回结果
    print('please wait...')
    if len(image_paths) == 1:
        img_url = encode_file(image_paths[0])
        rsp = VideoSynthesis.call(api_key=api_key,
                                  model='wan2.2-i2v-flash',
                                  prompt=prompt,
                                  img_url=img_url)
    else:
        assert len(image_paths) == 2:
        img_url = encode_file(image_paths[0])
        img2_url = encode_file(image_paths[1])
        rsp = VideoSynthesis.call(api_key=api_key,
                                  model='wan2.2-kf2v-flash',
                                  prompt=prompt,
                                  first_frame_url=img_url,
                                  last_frame_url=img2_url,)
    print(rsp)
    if rsp.status_code == HTTPStatus.OK:
        print("video_url:", rsp.output.video_url)
    else:
        print('Failed, status_code: %s, code: %s, message: %s' %
              (rsp.status_code, rsp.code, rsp.message))

    print(rsp.output.video_url)
    download_video(
        video_url=rsp.output.video_url,
        save_path=save_dir  # 用task_id命名，避免重复
    )

# ========== 新增：下载视频到本地 ==========
def download_video(video_url, save_path='./downloaded_video.mp4', chunk_size=1024*1024):
    """
    下载视频文件到本地
    :param video_url: 视频远程URL
    :param save_path: 本地保存路径（默认当前目录，文件名downloaded_video.mp4）
    :param chunk_size: 分块下载大小（默认1MB，避免内存占用过大）
    """
    try:
        # 发送GET请求（添加超时，避免无限等待）
        response = requests.get(video_url, stream=True, timeout=30)
        response.raise_for_status()  # 抛出HTTP错误（如404/500）

        # 获取文件总大小（可选，用于进度提示）
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0

        # 写入文件（分块下载，适合大文件）
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:  # 过滤空块
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    # 打印下载进度（可选）
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        print(f"下载进度: {progress:.1f}% ({downloaded_size}/{total_size} bytes)", end='\r')

        print(f"\n视频已成功保存到: {os.path.abspath(save_path)}")
        return save_path

    except requests.exceptions.RequestException as e:
        print(f"\n下载失败: {str(e)}")
        # 清理未下载完成的文件
        if os.path.exists(save_path):
            os.remove(save_path)
        return None

if __name__ == "__main__":
    # text2image("爱因斯坦拳打皮卡丘", "output.png")
    # image2image("将背景换成火焰山", "output.png", "modify.png")
    sample_call_i2v("爱因斯坦拳打皮卡丘", ["output.png"], "modify.mp4")