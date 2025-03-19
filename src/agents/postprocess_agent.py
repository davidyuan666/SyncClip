import logging
from threading import Thread
import os
import subprocess
from urllib.parse import urlparse
import torch
from src.utils.http_util import HttpUtil
from src.utils.tencent_cos_util import COSOperationsUtil
from src.utils.video_clip_util import VideoClipUtil

class PostprocessAgent:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.callback_url = None
        self.http_util = HttpUtil()
        self.video_clip_util = VideoClipUtil()
        self.project_dir = os.path.join(os.getcwd(), 'temp')
        self.cos_ops_util = COSOperationsUtil()


    def post_process_video(self, synthesized_final_video_url,
                    subtitle_path,
                    narration_audio_path,
                    preset,
                    project_no):
        """
        对视频进行后处理，包括生成字幕、添加旁白等
        
        Args:
            video_url (str): 原始视频URL
            selected_clips (list): 选中的视频片段列表，每个片段包含text字段
            preset (dict): 预设配置参数
            project_no (str): 项目编号
            
        Returns:
            dict: 包含处理结果的字典
        """
        try:
            print(f"Starting post-processing for project: {project_no}")
            
            # 2. 执行视频处理 - 修正参数传递
            raw_video_url, duration, narration_url, subtitle_url = self._process_video_in_detail(
                synthesized_final_video_url=synthesized_final_video_url,  # 参数名修正
                subtitle_path=subtitle_path,
                narration_audio_path=narration_audio_path,
                preset=preset,
                project_no=project_no
            )
            
            narration_audio_duration = self._get_audio_duration(narration_url)
            # 4. 构建返回结果
            result = {
                'raw_video_url': raw_video_url,
                'duration': duration,
                'narration_audio_url': narration_url,
                'narration_audio_duration': narration_audio_duration,
                'subtitle_url': subtitle_url
            }
            
            
            return result
                
        except Exception as e:
            error_msg = f"Failed in post-processing: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            if self.callback_url:
                Thread(target=self.http_util._send_callback, args=(self.callback_url, {
                        "data":{
                            "projectNo": project_no,
                        },
                        "error": {
                            "code": 500,
                            "message": error_msg
                        }})).start()
            raise Exception(error_msg)
        

    def _get_audio_duration(self, audio_path):
        """使用ffmpeg获取音频文件时长"""
        try:
            # Check if the input is a URL
            if audio_path.startswith('http'):
                # Create a temporary file to store the downloaded audio
                temp_dir = os.path.join(self.project_dir, 'temp_audio')
                os.makedirs(temp_dir, exist_ok=True)
                temp_audio_path = os.path.join(temp_dir, 'temp_audio.mp3')
                
                # Download the file if it's a URL
                parsed_url = urlparse(audio_path)
                object_key = parsed_url.path.lstrip('/')
                self.cos_ops_util.download_file(object_key, temp_audio_path)
                
                # Use the local path for ffprobe
                audio_path = temp_audio_path

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
            
            # Clean up temporary file if it was created
            if audio_path.startswith(self.project_dir):
                try:
                    os.remove(audio_path)
                except Exception as e:
                    self.logger.warning(f"Failed to clean up temporary audio file: {str(e)}")
            
            return duration
        except Exception as e:
            self.logger.error(f"Error getting audio duration: {str(e)}")
            raise


    

    '''
    后处理
    '''
    def _process_video_in_detail(self, synthesized_final_video_url,subtitle_path, narration_audio_path, preset, project_no):
        """
        对视频进行后处理,包括下载、处理、上传等操作
        
        Args:
            synthesized_final_video_url (str): 合成后的视频URL
            narration_content (str): 旁白内容
            narration_audio_path (str): 旁白音频路径
            preset (dict): 预设配置
            project_no (str): 项目编号
            
        Returns:
            tuple: (processed_video_url, video_duration, subtitle_url)
        """
        # 定义文件命名规则
        file_names = {
            'raw_video': f"{project_no}_raw.mp4",
            'processed_video': f"{project_no}_processed.mp4",
            'narration_audio': f"{project_no}_narration.mp3",
            'subtitle': f"{project_no}_subtitle.srt"
        }
        
        # 定义COS存储路径结构
        cos_paths = {
            'video': {
                'raw': f"{project_no}/raw/{file_names['raw_video']}",
                'processed': f"{project_no}/processed/{file_names['processed_video']}"
            },
            'audio': {
                'narration': f"{project_no}/audio/{file_names['narration_audio']}"
            },
            'subtitle': {
                'srt': f"{project_no}/subtitle/{file_names['subtitle']}"
            }
        }
        
        # 创建临时工作目录
        temp_dir = os.path.join(self.project_dir, 'post_process', project_no)
        os.makedirs(temp_dir, exist_ok=True)
        
        # 初始化临时文件路径
        temp_files = {
            'raw_video': os.path.join(temp_dir, file_names['raw_video']),
            'processed_video': os.path.join(temp_dir, file_names['processed_video']),
            'narration_audio': narration_audio_path if narration_audio_path else None,
            'subtitle': subtitle_path if subtitle_path else None
        }
        
        try:
            # 验证输入参数
            if not synthesized_final_video_url:
                raise ValueError("No synthesized video URL provided")
                
            # 1. 下载原始视频
            self.logger.info(f"Downloading merged video from COS: {synthesized_final_video_url}")
            parsed_url = urlparse(synthesized_final_video_url)
            object_key = parsed_url.path.lstrip('/')
            
            self.cos_ops_util.download_file(
                object_key,
                temp_files['raw_video']
            )
            
            if not os.path.exists(temp_files['raw_video']):
                raise FileNotFoundError(f"Downloaded video file not found at: {temp_files['raw_video']}")
                
            self.logger.info(f"Raw video downloaded to: {temp_files['raw_video']}")
            
               # 清理 GPU 缓存
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
                
            # 获取可用 GPU 内存
            if torch.cuda.is_available():
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                self.logger.info(f"Available GPU memory: {free_memory / 1024**2:.2f} MB")
            

            # 3. 应用视频效果
            # processed_video_path, video_duration = (
            #     self.video_clip_util.apply_effects(
            #         video_path=temp_files['raw_video'],
            #         subtitle_path=temp_files['subtitle'],
            #         narration_audio_path=temp_files['narration_audio'],
            #         preset=preset,
            #         project_no=project_no
            #     )
            # )

            # if not processed_video_path:
            #     raise ValueError("Video processing failed: no output path returned")
            
            # temp_files['processed_video'] = processed_video_path
            
            # 4. 验证处理后的文件
            # if not os.path.exists(temp_files['processed_video']):
            #     raise FileNotFoundError(f"Processed video not found at: {temp_files['processed_video']}")
                


            video_path=temp_files['raw_video']
            video_duration = self.video_clip_util.get_video_duration(video_path)

            
     
            # 5. 上传处理后的文件并获取URLs
            urls = {}
            
            # 上传处理后的视频
            self.cos_ops_util.upload_file(
                temp_files['raw_video'],
                cos_paths['video']['raw']
            )
            urls['video'] = self.cos_ops_util.get_file_url(
                cos_paths['video']['raw']
            )
            
            # 上传旁白音频(如果存在)
            urls['narration'] = None
            if temp_files['narration_audio'] and os.path.exists(temp_files['narration_audio']):
                self.cos_ops_util.upload_file(
                    temp_files['narration_audio'],
                    cos_paths['audio']['narration']
                )
                urls['narration'] = self.cos_ops_util.get_file_url(
                    cos_paths['audio']['narration']
                )
            
            # 上传字幕文件(如果存在)
            urls['subtitle'] = None
            if temp_files['subtitle'] and os.path.exists(temp_files['subtitle']):
                self.cos_ops_util.upload_file(
                    temp_files['subtitle'],
                    cos_paths['subtitle']['srt']
                )
                urls['subtitle'] = self.cos_ops_util.get_file_url(
                    cos_paths['subtitle']['srt']
                )
            
            return (
                urls['video'],
                video_duration,
                urls['narration'],
                urls['subtitle']
            )
            
        except Exception as e:
            error_msg = f"Error in post-processing video: {str(e)}"
            self.logger.error(error_msg)
            self.http_util._send_callback(self.callback_url, {
                "data":{
                    "projectNo": project_no,
                },

                "error": {
                    "code": 500,
                    "message": error_msg
                }
            })
            raise
            
        finally:
            # 清理临时文件
            for file_path in temp_files.values():
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        self.logger.debug(f"Cleaned up temporary file: {file_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to clean up file {file_path}: {str(e)}")
            