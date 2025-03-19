import logging
import requests
from threading import Thread
import ffmpeg
from src.utils.http_util import HttpUtil
from src.utils.tencent_cos_util import COSOperationsUtil
from src.utils.common_util import CommonUtil
from urllib.parse import urlparse
import time
from src.agents.blip_agent import BlipAgent
import os
import uuid
from PIL import Image
import cv2

class VideoAgent:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.callback_url = None
        self.http_util = HttpUtil()
        self.cos_ops_util = COSOperationsUtil()
        self.common_util = CommonUtil()
        self.blip_agent = BlipAgent()
        self.desired_frames = int(os.getenv('DESIRED_FRAMES'))
        self.interval_seconds = int(os.getenv('INTERVAL_SECONDS'))
        self.logger = logging.getLogger(__name__)



    def get_video_captions(self, video_url, project_no):
        """获取视频字幕"""
        try:

            self.logger.info(f"video_url: {video_url}")
            self.logger.info(f"project_no: {project_no}")
            
            # 获取视频时长
            try:
                # 首先尝试直接从URL获取时长
                try:
                    probe = ffmpeg.probe(video_url, cmd='ffprobe')
                    duration = float(probe['streams'][0]['duration'])
                except Exception as e:
                    self.logger.error(f"Failed to probe remote video: {e}")
                    # 如果直接探测失败，尝试使用HTTP HEAD请求获取Content-Length
                    response = requests.head(video_url)
                    if 'Content-Duration' in response.headers:
                        duration = float(response.headers['Content-Duration'])
                    else:
                        raise Exception("Could not determine video duration")
                        
                self.logger.info(f"Video duration: {duration} seconds")
                
            except Exception as e:
                self.logger.error(f"Error getting video duration, using default interval: {str(e)}")
                if self.callback_url:
                    Thread(target=self.http_util._send_callback, args=(self.callback_url, {
                            "data":{
                                "projectNo": project_no,
                            },
                            "error": {
                                "code": 500,
                                "message": "Error getting video duration"
                            }})).start()
            
            captions = self._get_video_captions_by_local(
                video_url=video_url,
                project_no=project_no
            )
            
            # 保存字幕数据
            self.common_util.save_captions_to_json(project_no, captions)
            return captions
            
        except Exception as e:
            self.logger.error(f"Error getting video captions: {str(e)}")
            raise


    
    def _get_video_captions_by_local(self, video_url, project_no):
        """
        本地处理视频帧并生成描述
        
        Args:
            video_url (str): 视频URL
            project_no (str): 项目编号
                
        Returns:
            dict: 包含帧描述和时间戳的响应
        """
        try:
            if video_url is None:
                raise ValueError("Video URL is required")

            # 从 URL 中提取 COS 文件路径
            if 'https' in video_url:
                parsed_url = urlparse(video_url)
                cos_file_path = parsed_url.path.lstrip('/')
                if cos_file_path.startswith(self.cos_ops_util.bucket_name):
                    cos_file_path = cos_file_path[len(self.cos_ops_util.bucket_name):].lstrip('/')
            else:
                cos_file_path = video_url

            self.logger.info(f"Processing video captions for COS path: {cos_file_path}")

            # 使用 process_all_frames 提取帧
            frame_results = self.process_all_frames(
                project_no=project_no,
                video_url=video_url
            )

            if frame_results is None:
                raise Exception("Failed to extract video frames")

    

            # 处理每一帧并生成描述
            captioned_frames = []
            total_frames = len(frame_results['frames'])
            
            self.logger.info(f"=== Starting caption generation for {total_frames} frames ===")
            start_time = time.time()

            for idx, frame_info in enumerate(frame_results['frames'], 1):
                self.logger.info(f"Processing frame {idx}/{total_frames}")
                self.logger.info(f"Timestamp: {frame_info['timestamp']}")
                
                # 生成图片描述
                caption = self.blip_agent.generate_caption_by_local_path(
                    frame_info['local_frame_path']
                )
                
                captioned_frames.append({
                    'frame_url': frame_info['local_frame_path'],  # 保持与原API响应格式一致
                    'timestamp': frame_info['timestamp'],
                    'caption': caption
                })

                # 如果达到所需的帧数，就停止处理
                if idx >= self.desired_frames:
                    break

            elapsed_time = time.time() - start_time
            self.logger.info(f"=== Caption generation completed ===")
            self.logger.info(f"Total frames processed: {len(captioned_frames)}")
            self.logger.info(f"Total time taken: {elapsed_time:.2f} seconds")
            self.logger.info(f"Average time per frame: {elapsed_time/len(captioned_frames):.2f} seconds")

            return {
                "success": True,
                "video_url": video_url,
                "cos_file_path": cos_file_path,
                "total_frames": len(captioned_frames),
                "frames": captioned_frames
            }

        except Exception as e:
            error_msg = f"Error getting video captions: {str(e)}"
            self.logger.error(error_msg)
            if self.callback_url:
                Thread(target=self.http_util._send_callback, args=(self.callback_url, {
                        "data":{
                            "projectNo": project_no,
                        },
                        "error": {
                            "code": 500,
                            "message": error_msg
                        }})).start()
            raise



    '''
    处理每一帧
    '''
    def process_all_frames(self, project_no, video_url):
        """按固定时间间隔提取视频帧
        Args:
            project_no: 项目编号
            video_url: 视频URL
            local_video_path: 视频本地路径（如果已有）
        Returns:
            dict: 包含视频信息和帧信息的字典
                {
                    "local_video_path": str,  # 视频本地路径
                    "frames": list[dict],     # 帧信息列表，每个元素包含local_frame_path和timestamp
                    "video_info": dict        # 视频基本信息（fps, duration等）
                }
        """
        try:
            local_video_path = self.http_util.download_video_from_cos(video_url, project_no)
            if not local_video_path:
                raise ValueError("Failed to download video from COS")
                
            # Create output directory for frames
            frames_dir = os.path.join(os.getcwd(), 'temp', project_no, 'frames')
            os.makedirs(frames_dir, exist_ok=True)

            # Open video file
            cap = cv2.VideoCapture(local_video_path)
            if not cap.isOpened():
                raise ValueError("Failed to open video file")

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = int(total_frames / fps)  # 视频总时长(秒)

            # 计算采样时间点（秒）
            sample_timestamps = range(0, duration, self.interval_seconds)
            
            frames = []
            
            for timestamp in sample_timestamps:
                # 计算对应的帧位置
                frame_pos = int(timestamp * fps)
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = cap.read()
                if not ret:
                    continue

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Save frame locally
                frame_filename = f"frame_{uuid.uuid4()}.jpg"
                frame_path = os.path.join(frames_dir, frame_filename)
                
                Image.fromarray(frame_rgb).save(frame_path, quality=60)
                
                frames.append({
                    "local_frame_path": frame_path,
                    "timestamp": timestamp
                })
            
            cap.release()
            
            return {
                "local_video_path": local_video_path,
                "frames": frames,
                "video_info": {
                    "fps": fps,
                    "duration": duration,
                    "total_frames": total_frames
                }
            }

        except Exception as e:
            self.logger.error(f"Error processing frames: {str(e)}")
            return None

