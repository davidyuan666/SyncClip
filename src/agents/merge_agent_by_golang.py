import os
from src.utils.video_clip_util import VideoClipUtil
from threading import Thread
import logging
from src.utils.tencent_cos_util import COSOperationsUtil
from src.utils.http_util import HttpUtil
import ffmpeg
from tqdm import tqdm
import subprocess
import time  # 确保在文件顶部导入
import uuid
from urllib.parse import urlparse
import requests
import json
from typing import Dict, Union
import shutil


class MergeGoAgent:
    def __init__(self):
        self.cos_bucket_name = os.getenv('COS_BUCKET_NAME')
        self.cos_ops_util = COSOperationsUtil()
        self.video_clip_util = VideoClipUtil()
        self.http_util = HttpUtil()
        self.blip_video_captions_api_url = os.getenv('BLIP_VIDEO_CAPTIONS_API_URL')
        self.synthesize_video_api_url = os.getenv('SYNTHESIZE_VIDEO_API_URL')
        self.merge_dir = os.path.join(os.getcwd(), 'merge')
        self.base_cos_url = os.getenv('BASE_COS_URL')
        self.project_dir = os.path.join(os.getcwd(), 'temp')
        self.logger = logging.getLogger(__name__)

    '''
    入口函数
    '''
    async def process_merge(self, data):
        """
        处理合并请求
        
        Args:
            data (dict): 包含合并参数的字典
        """
        try:
            # 检查必要的键是否存在
            required_keys = ['projectNo', 'preset', 'clips', 'callbackUrl']
            for key in required_keys:
                if key not in data:
                    raise ValueError(f"Missing required key: {key}")

            self.logger.info(f'================================> process merge data: {data}')

            project_no = data['projectNo']
            preset = data['preset']
            clips = data['clips']
            self.callback_url = data['callbackUrl']
            narration = data.get('narration', '')

            # 清理目录
            self._clean_project_directories(project_no)

            # 准备输出目录
            output_dir = os.path.join(self.merge_dir, project_no)
            os.makedirs(output_dir, exist_ok=True)

            # 分离视频和转场效果
            videos = []
            for clip in clips:
                if clip['type'] == 'video':
                    # 从URL中提取文件名
                    # video_local_path = self._download_video_from_cos(clip['videoUrl'],project_no)
                    # 下载文件
                    videos.append({
                        "type": "video",
                        "path": clip['videoUrl'],
                        "effect": 0,
                        "duration": clip.get('duration', 0)
                    })
                elif clip['type'] == 'effect':
                    videos.append({
                        "type": "effect",
                        "path": "",
                        "effect": clip.get('effect', 0),
                        "duration": 0
                    })

            # 准备请求参数
            merge_params = {
                "project_no": project_no,
                "output_dir": project_no,
                "videos": videos,
                "bgm": preset.get('bgm', 0)
            }

            
            narration_audio_path = self.video_clip_util._generate_audio(narration, preset, project_no)
           
                           # Upload narration audio
            narration_filename = os.path.basename(narration_audio_path)
            narration_remote_path = f"{project_no}/audio/{str(uuid.uuid4())}_{narration_filename}"
            self.cos_ops_util.upload_file(
                local_file_path=narration_audio_path,
                cos_file_path=narration_remote_path
            )
            narration_audio_url = self.cos_ops_util.get_file_url(narration_remote_path)
            
            merge_params['audio'] = {
                            "path": narration_audio_url
                        }
           
           
           
            # 添加音频配置（如果存在）
            if narration_audio_path:
                merge_params['audio'] = {"path": narration_audio_path}
            else:
                error_data = {
                    "error": {
                        "code": 500,
                        "message": f"narration is empty and no audio file generated"
                    }
                }
                
                if hasattr(self, 'callback_url'):
                    Thread(target=self.http_util._send_callback, args=(self.callback_url, error_data)).start()

                merge_params['audio'] = {"path": ""}

            if 'subtitle' in preset:
                local_subtitle_path = self._generate_subtitles(narration_audio_path, narration)
                # Upload subtitle
                subtitle_filename = os.path.basename(local_subtitle_path)
                subtitle_remote_path = f"{project_no}/subtitle/{str(uuid.uuid4())}_{subtitle_filename}"
                self.cos_ops_util.upload_file(
                    local_file_path=local_subtitle_path,
                    cos_file_path=subtitle_remote_path
                )
                subtitle_url = self.cos_ops_util.get_file_url(subtitle_remote_path)
                
                
                if local_subtitle_path:
                    merge_params['subtitle'] = {
                            "font": preset.get('subtitleFont', 1),
                            "style": preset.get('subtitleStyle', 1),
                            "color": preset.get('subtitleColor', '#FFFFFF'),
                            "path": subtitle_url  # 使用本地路径
                        }
                else:
                    error_data = {
                        "error": {
                            "code": 500,
                            "message": f"subtitle is empty and no subtitle file generated"
                        }
                    }

                    if hasattr(self, 'callback_url'):
                        Thread(target=self.http_util._send_callback, args=(self.callback_url, error_data)).start()

                    merge_params['subtitle'] = {"path": ""}


            # 添加水印配置（如果存在）
            if 'watermark' in preset:
                merge_params['watermark'] = {
                    "text": preset.get('watermarkText', ''),
                    "position": preset.get('watermarkPosition', 4)
                }

            # 发送合并请求
            self.logger.info('[step 1]: Sending merge request')
            result = self.send_merge_request(**merge_params)

            if "error" in result:
                raise Exception(result["error"]["message"])

            processed_video_path = result["data"]["path"]
            
            # 上传到 COS
            self.logger.info('[step 2]: Uploading merged video to COS')
            upload_response = self.upload_video_to_cos(processed_video_path, project_no)
            if upload_response is None:
                raise ValueError("Upload response is None")
            
            merged_video_url = upload_response.get('video_url', '')
            if not merged_video_url:
                raise ValueError("No video URL in upload response")

            # 获取视频时长
            try:
                probe = ffmpeg.probe(processed_video_path)
                video_duration = float(probe['streams'][0]['duration'])
            except Exception as e:
                self.logger.error(f"Error getting video duration: {str(e)}")
                video_duration = 0

            # 准备回调数据
            callback_data = {
                "data": {
                    "projectNo": project_no,
                    "videoUrl": merged_video_url,
                    "duration": int(video_duration)
                }
            }

            # 发送回调
            Thread(target=self.http_util._send_callback, args=(self.callback_url, callback_data)).start()

            self.logger.info(f'====> callback data is:\n {callback_data}')
            return callback_data

        except Exception as e:
            self.logger.error(f'error is: {str(e)}')
            error_data = {
                "error": {
                    "code": 500,
                    "message": f"Internal server error during merge process: {str(e)}"
                }
            }
            
            if hasattr(self, 'callback_url'):
                Thread(target=self.http_util._send_callback, args=(self.callback_url, error_data)).start()

            return error_data
        
    def _clean_project_directories(self, project_no: str):
        """
        清理项目相关的临时目录和合并目录
        
        Args:
            project_no (str): 项目编号
        """
        try:
            # 清理merge目录
            merge_project_dir = os.path.join(self.merge_dir, project_no)
            if os.path.exists(merge_project_dir):
                self.logger.info(f"Cleaning merge directory: {merge_project_dir}")
                for item in os.listdir(merge_project_dir):
                    item_path = os.path.join(merge_project_dir, item)
                    try:
                        if os.path.isfile(item_path):
                            os.unlink(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                    except Exception as e:
                        self.logger.warning(f"Error while deleting {item_path}: {str(e)}")

            # 清理temp目录
            temp_project_dir = os.path.join(self.project_dir, project_no)
            if os.path.exists(temp_project_dir):
                self.logger.info(f"Cleaning temp directory: {temp_project_dir}")
                for item in os.listdir(temp_project_dir):
                    item_path = os.path.join(temp_project_dir, item)
                    try:
                        if os.path.isfile(item_path):
                            os.unlink(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                    except Exception as e:
                        self.logger.warning(f"Error while deleting {item_path}: {str(e)}")

            # 重新创建目录
            os.makedirs(merge_project_dir, exist_ok=True)
            os.makedirs(temp_project_dir, exist_ok=True)

            self.logger.info(f"Successfully cleaned project directories for project: {project_no}")

        except Exception as e:
            self.logger.error(f"Error cleaning project directories: {str(e)}")
            raise Exception(f"Failed to clean project directories: {str(e)}")
        
        
    def _generate_subtitles(self, narration_audio_path, narration_content):
        """生成 SRT 格式字幕文件"""
        try:
            subtitle_path = os.path.join(self.project_dir, f"subtitle_{int(time.time())}.srt")
            
            # 首先尝试使用 LLM 获取音频分段
            try:
                audio_segments = self.video_clip_util._get_audio_segments(narration_audio_path)
                if audio_segments:
                    self.logger.info("Using LLM audio segments for subtitle generation")
                    return self._generate_subtitles_from_segments(subtitle_path, audio_segments)
            except Exception as e:
                self.logger.warning(f"LLM audio segmentation failed: {str(e)}, falling back to traditional method")
            
            # 降级：使用传统方法
            self.logger.info("Falling back to traditional subtitle generation method")
            return self._generate_subtitles_traditional(subtitle_path, narration_audio_path,narration_content)
            
        except Exception as e:
            self.logger.error(f"Error in subtitle generation: {str(e)}")
            raise
            

    def _format_time(self, seconds):
        """将秒数转换为 SRT 时间格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"
    

    def _get_audio_duration(self, audio_path):
        """使用ffmpeg获取音频文件时长"""
        try:
            cmd = [
                'ffprobe', 
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                audio_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"FFprobe error: {result.stderr}")
                
            duration = float(result.stdout.strip())
            return duration
        except Exception as e:
            self.logger.error(f"Error getting audio duration: {str(e)}")
            raise


    def _generate_subtitles_from_segments(self, subtitle_path, audio_segments):
        """使用 LLM 音频分段生成字幕"""
        try:
            with open(subtitle_path, 'w', encoding='utf-8') as f:
                for i, segment in enumerate(audio_segments, 1):
                    start_str = self._format_time(segment['start'])
                    end_str = self._format_time(segment['end'])
                    text = segment['text'].strip()
                    
                    # 写入 SRT 格式
                    f.write(f"{i}\n")
                    f.write(f"{start_str} --> {end_str}\n")
                    f.write(f"{text}\n\n")
                    
            self.logger.info(f"Generated subtitles using LLM segments: {len(audio_segments)} segments")
            return subtitle_path
            
        except Exception as e:
            self.logger.error(f"Error in LLM subtitle generation: {str(e)}")
            raise
            
    def _generate_subtitles_traditional(self, subtitle_path, narration_audio_path,narration_content):
        """使用传统方法生成字幕"""
        try:
            # 将旁白内容按句子分割
            sentences = narration_content.split('。')
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # 估算每句话的时间戳
            total_duration = self._get_audio_duration(narration_audio_path)
            avg_duration = total_duration / len(sentences)
            
            with open(subtitle_path, 'w', encoding='utf-8') as f:
                current_time = 0
                for i, sentence in enumerate(sentences, 1):
                    # 根据句子长度调整持续时间
                    duration = min(avg_duration * (len(sentence) / 20), avg_duration * 1.5)  # 20个字为基准
                    
                    start_time = current_time
                    end_time = start_time + duration
                    
                    # 转换时间格式
                    start_str = self._format_time(start_time)
                    end_str = self._format_time(end_time)
                    
                    # 写入 SRT 格式
                    f.write(f"{i}\n")
                    f.write(f"{start_str} --> {end_str}\n")
                    f.write(f"{sentence}\n\n")
                    
                    current_time = end_time + 0.1  # 添加小间隔
                    
            self.logger.info(f"Generated subtitles using traditional method: {len(sentences)} segments")
            return subtitle_path
            
        except Exception as e:
            self.logger.error(f"Error in traditional subtitle generation: {str(e)}")
            raise


    def upload_video_to_cos(self, local_video_path, project_no):
        """Upload merged video to COS
        
        Args:
            video_path (str): Local path to the video file
            project_no (str): Project identifier
            
        Returns:
            str: Public URL of the uploaded video
        """
        try:
            if not os.path.exists(local_video_path):
                raise FileNotFoundError(f"Local Video file not found: {local_video_path}")
            
            # Create the COS path with project number, UUID, and 'merged' folder
            filename = os.path.basename(local_video_path)
            remote_cos_path = f"{project_no}/merged/{str(uuid.uuid4())}_{filename}"
            
            self.logger.info(f"Starting upload of {local_video_path} to COS...")
            
            # Upload the video to COS
            self.cos_ops_util.upload_file(
                local_file_path=local_video_path,
                cos_file_path=remote_cos_path
            )
            
            video_url = self.cos_ops_util.get_file_url(remote_cos_path)

            self.logger.info(f"Successfully uploaded merged video to COS: {remote_cos_path}")
            
            return {
                "video_url": video_url,
                "project_no": project_no
            }
            
        except Exception as e:
            self.logger.error(f"Error uploading merged video {local_video_path} to COS: {str(e)}")
            return None


    
    def _download_video_from_cos(self, cos_video_url, project_no):
        try:
            print(f'cos video url: {cos_video_url}')

            # Extract remote path from URL
            if 'https' in cos_video_url:
                parsed_url = urlparse(cos_video_url)
                remote_path = parsed_url.path.lstrip('/')
                if remote_path.startswith(self.cos_ops_util.bucket_name):
                    remote_path = remote_path[len(self.cos_ops_util.bucket_name):].lstrip('/')
            else:
                remote_path = cos_video_url

            # Generate a shorter filename using UUID
            file_extension = os.path.splitext(remote_path)[1]  # Get original extension
            short_filename = f"{uuid.uuid4().hex[:8]}{file_extension}"  # Use first 8 chars of UUID
            
            # Create temp directory path
            temp_dir = os.path.join(os.getcwd(), 'temp', project_no)
            video_local_path = os.path.join(temp_dir, short_filename)

            # Ensure the directory exists
            os.makedirs(temp_dir, exist_ok=True)

            # Download the file
            print(f"\033[93mDownloading {remote_path}\033[0m")
            try:
                self.cos_ops_util.download_file(
                    remote_path,
                    video_local_path
                )
            except Exception as e:
                print(f"Failed to download from COS: {str(e)}")
                raise
                
            if not os.path.exists(video_local_path):
                raise FileNotFoundError(f"File download failed - file not found at {video_local_path}")
            
            if os.path.getsize(video_local_path) == 0:
                raise ValueError(f"Downloaded file is empty: {video_local_path}")

            print(f"Successfully downloaded video to {video_local_path}")
            return video_local_path
                
        except Exception as e:
            print(f"\033[93mError downloading video from {cos_video_url}: {str(e)}\033[0m")
            return None
        

    

    def send_merge_request(
        self,
        project_no: str,
        output_dir: str,
        videos: list,
        audio: Dict = None,
        bgm: int = 0,
        subtitle: Dict = None,
        watermark: Dict = None,
        api_url: str = os.getenv('GO_MERGE_API_URL', 'http://127.0.0.1:8009/api/v1/merge')
    ) -> Dict:
        """
        发送视频合并请求到指定的API端点
        
        Args:
            project_no (str): 项目编号
            output_dir (str): 输出目录路径
            videos (list): 视频和转场效果列表
            audio (dict, optional): 旁白音频信息
            bgm (int, optional): 背景音乐ID，默认为0
            subtitle (dict, optional): 字幕信息
            watermark (dict, optional): 水印信息
            api_url (str, optional): API端点URL
            
        Returns:
            dict: API响应数据，成功时返回 {"data": {"path": "输出路径"}}，
                失败时返回 {"error": {"code": 错误码, "message": "错误信息"}}
        """
        try:
            # 构建请求数据
            payload = {
                "project_no": project_no,
                "output_dir": output_dir,
                "videos": videos,
                "bgm": bgm
            }
            
            # 添加可选参数
            if audio:
                payload["audio"] = audio
            if subtitle:
                payload["subtitle"] = subtitle
            if watermark:
                payload["watermark"] = watermark
                
            # 发送POST请求
            logging.info(f"Sending merge request to {api_url}")
            logging.info(f"Request payload: {json.dumps(payload, indent=2)}")
            
            response = requests.post(
                api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=3600  # 设置1小时超时，因为视频处理可能需要较长时间
            )
            
            # Check response status code
            response.raise_for_status()
            # 解析响应数据
            result = response.json()
            logging.info(f"Merge request successful: {result}")
            return result
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Network error during merge request: {str(e)}"
            logging.error(error_msg)
            return {
                "error": {
                    "code": 500,
                    "message": error_msg
                }
            }
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse API response: {str(e)}"
            logging.error(error_msg)
            return {
                "error": {
                    "code": 500,
                    "message": error_msg
                }
            }
        except Exception as e:
            error_msg = f"Unexpected error during merge request: {str(e)}"
            logging.error(error_msg)
            return {
                "error": {
                    "code": 500,
                    "message": error_msg
                }
            }
        
        