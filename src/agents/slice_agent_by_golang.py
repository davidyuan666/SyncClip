import logging
import os
import json
from threading import Thread
import time  # 确保在文件顶部导入
from typing import Dict, Any
import subprocess
from src.utils.video_clip_util import VideoClipUtil
from src.utils.tencent_cos_util import COSOperationsUtil
from src.utils.http_util import HttpUtil
from src.utils.common_util import CommonUtil
from src.agents.blip_agent import BlipAgent
from src.agents.combine_agent import CombineAgent
from src.agents.video_agent import VideoAgent
from src.agents.frame_agent import FrameAgent
from src.agents.segment_agent import SegmentAgent
from src.agents.clip_agent import ClipAgent
from src.agents.audio_agent import AudioAgent
from src.agents.synthesize_agent import SynthesizeAgent
from src.agents.postprocess_agent import PostprocessAgent
import shutil
import requests
import uuid
from urllib.parse import urlparse
import ffmpeg

class SliceGoAgent:
    def __init__(self):
        # 获取 BLIP 处理器实例
        self.video_clip_util = VideoClipUtil()
        self.cos_ops_util = COSOperationsUtil()
        self.http_util = HttpUtil()
        self.common_util = CommonUtil()
        self.cos_bucket_name = os.getenv('COS_BUCKET_NAME')
        self.base_cos_url = os.getenv('BASE_COS_URL')
        self.desired_frames = int(os.getenv('DESIRED_FRAMES'))
        self.interval_seconds = int(os.getenv('INTERVAL_SECONDS'))
        self.status_callback_url = os.getenv('STATUS_CALLBACK_URL')
        self.project_dir = os.path.join(os.getcwd(), 'temp')
        self.slice_dir = os.path.join(os.getcwd(), 'slice')
        self.merge_dir = os.path.join(os.getcwd(), 'merge')
        self.logger = logging.getLogger(__name__)
        self.callback_url = None
        if not self._verify_ffmpeg_installation():
            self.logger.warning("FFmpeg CUDA support not available, using CPU processing")
    
        self.combine_agent = CombineAgent()
        self.video_agent = VideoAgent()
        self.frame_agent = FrameAgent()
        self.segment_agent = SegmentAgent()
        self.clip_agent = ClipAgent()
        self.audio_agent = AudioAgent()
        self.synthesize_agent = SynthesizeAgent()
        self.postprocess_agent = PostprocessAgent()
        self.blip_agent = BlipAgent()

    async def process_slice(self, data: Dict[str, Any]):
        """处理视频的主要步骤"""
        process_start_time = time.time()

        self.logger.info(f'================================> process slice data: {data}')
        project_no = data['projectNo']
        materials = data['materials']
        preset = data['preset']
        self.callback_url = data['callbackUrl']

        # 清理目录
        self._clean_project_directories(project_no)

        # 扩展步骤计时记录
        steps_timing = {
            'merge_videos': 0,
            'get_captions': 0,
            'process_segments': 0,
            'merge_segments': 0,
            'clip_strategy': 0,
            'synthesize_video': 0,
            'post_process': 0,
            'total_time': 0,
            'sync_audio_with_video': 0
        }
        
        try:
            # 1. 合并视频
            self.logger.info(f"[Step 1/9] Merging input videos for project {project_no}")
            start_time = time.time()
            try:
                merged_input_video_url = self.combine_agent.combine_raw_videos(
                    materials, 
                    project_no
                )
                self.logger.info(f"merged_input_video_url: {merged_input_video_url}")
                steps_timing['merge_videos'] = int((time.time() - start_time))
                self.logger.info(f"Video merging completed in {steps_timing['merge_videos']:.2f} seconds")
            except Exception as e:
                raise Exception(f"Failed to merge videos: {str(e)}")
            
            # 2. 获取字幕
            self.logger.info(f"[Step 2/9] Getting video captions")
            start_time = time.time()
            try:
                captions = self.video_agent.get_video_captions(
                    merged_input_video_url,
                    project_no
                )
                steps_timing['get_captions'] = int((time.time() - start_time))
                self.logger.info(f"Caption generation completed in {steps_timing['get_captions']:.2f} seconds")
            except Exception as e:
                raise Exception(f"Failed to get video captions: {str(e)}")
            
            # 3. 处理字幕和片段
            self.logger.info(f"[Step 3/9] Processing segments")
            start_time = time.time()
            try:
                segments = self.frame_agent.process_segments(
                    captions,
                    merged_input_video_url,
                    project_no
                )
                steps_timing['process_segments'] = int((time.time() - start_time))
                self.logger.info(f"Segment processing completed in {steps_timing['process_segments']:.2f} seconds")
            except Exception as e:
                raise Exception(f"Failed to process segments: {str(e)}")
            
            # 4. 合并相似片段
            self.logger.info(f"[Step 4/9] Merging similar segments")
            start_time = time.time()
            try:
                merged_segments = self.segment_agent.merge_similar_segments(
                    segments,
                    preset.get('narrationLang'),
                    project_no
                )
                self.logger.info(f"merged_segments: {merged_segments}")
                steps_timing['merge_segments'] = int((time.time() - start_time))
                self.logger.info(f"Segment merging completed in {steps_timing['merge_segments']:.2f} seconds")
            except Exception as e:
                raise Exception(f"Failed to merge segments: {str(e)}")
            
            # 5. 应用剪辑策略
            self.logger.info(f"[Step 5/9] Applying clip strategy")
            start_time = time.time()
            try:
                selected_clips = self.clip_agent.apply_clip_strategy(
                    merged_segments,
                    preset,
                    project_no
                )
                '''
                selected_clips: [{'end': 31.0, 'start': 28.0, 'text': 'a person holding a sandwich with meat and cheese'}, {'end': 33.0, 'start': 31.0, 'text': 'a person holding a pastry with a bite taken out of it'}, {'end': 37.0, 'start': 35.0, 'text': 'a bakery with a large display case of food'}, {'end': 49.0, 'start': 47.0, 'text': 'a menu of a japanese restaurant with a variety of food items'}, {'end': 145.0, 'start': 140.0, 'text': 'a display case with lots of different kinds of candy and cakes'}]
                '''
                self.logger.info(f"selected_clips: {selected_clips}")

                steps_timing['clip_strategy'] = int((time.time() - start_time))
                self.logger.info(f"Clip strategy completed in {steps_timing['clip_strategy']:.2f} seconds")
            except Exception as e:
                raise Exception(f"Failed to apply clip strategy: {str(e)}")
            


            self.logger.info(f"[Step 6/9] Sync Audio with Video")
            start_time = time.time()
            try:
                narration_audio_path, sync_selected_clips, metadata_result, subtitle_path = self.audio_agent.sync_audio_with_video(
                    selected_clips,
                    preset,
                    project_no
                )
                self.logger.info(f"sync_selected_clips: {sync_selected_clips}")
                steps_timing['sync_audio_with_video'] = int((time.time() - start_time))
                self.logger.info(f"Audio synchronization completed in {steps_timing['sync_audio_with_video']:.2f} seconds")
            except Exception as e:
                raise Exception(f"Failed to sync audio with video: {str(e)}")
            
            # 6. 合成最终视频
            self.logger.info(f"[Step 7/9] Synthesizing final video")
            start_time = time.time()
            try:
                synthesized_final_video_url, clip_desc = self.synthesize_agent.synthesize_final_video(
                    sync_selected_clips,
                    merged_input_video_url,
                    project_no
                )
                steps_timing['synthesize_video'] = int((time.time() - start_time))
                self.logger.info(f"Video synthesis completed in {steps_timing['synthesize_video']:.2f} seconds")
            except Exception as e:
                raise Exception(f"Failed to synthesize final video: {str(e)}")


            self.logger.info(f"原始的合成视频: {synthesized_final_video_url}")
            self.logger.info(f"原始的视频切片: {clip_desc}")

            self.logger.info(f"[Step 8/9] Sending merge request")
            videos = []
            for clip in clip_desc:
                # video_local_path = self._download_video_from_cos(clip.get('url', ''),project_no)
                videos.append({
                        "type": "video",
                        "path": clip.get('url', ''),
                        "effect": 0,
                        "duration": clip.get('duration', 0)
                    })
  
            # 准备输出目录
            output_dir = os.path.join(self.slice_dir, project_no)
            os.makedirs(output_dir, exist_ok=True)
            # 准备请求参数
            merge_params = {
                "project_no": project_no,
                "output_dir": project_no,
                "videos": videos,
                "bgm": preset.get('bgm', 0)
            }

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


            # Upload subtitle
            subtitle_filename = os.path.basename(subtitle_path)
            subtitle_remote_path = f"{project_no}/subtitle/{str(uuid.uuid4())}_{subtitle_filename}"
            self.cos_ops_util.upload_file(
                local_file_path=subtitle_path,
                cos_file_path=subtitle_remote_path
            )
            subtitle_url = self.cos_ops_util.get_file_url(subtitle_remote_path)
            
            if 'subtitle' in preset:
                merge_params['subtitle'] = {
                            "font": preset.get('subtitleFont', 1),
                            "style": preset.get('subtitleStyle', 1),
                            "color": preset.get('subtitleColor', '#FFFFFF'),
                            "path": subtitle_url  # 使用本地路径
                        }


            # 添加水印配置（如果存在）
            if 'watermark' in preset:
                merge_params['watermark'] = {
                    "text": preset.get('watermarkText', ''),
                    "position": preset.get('watermarkPosition', 4)
                }

            result = self.send_merge_request(**merge_params)

            if "error" in result:
                raise Exception(result["error"]["message"])

            processed_video_path = result["data"]["path"]
            
            # 上传到 COS
            self.logger.info('[step 7/9]: Uploading merged video to COS')
            upload_response = self.upload_video_to_cos(processed_video_path, project_no)
            if upload_response is None:
                raise ValueError("Upload response is None")
            
            merged_video_url = upload_response.get('video_url', '')
            if not merged_video_url:
                raise ValueError("No video URL in upload response")

            self.logger.info(f"最终合成的视频: {merged_video_url}")
            # 8. 准备结果
            self.logger.info(f"[Step 8/9] Sending Callback result")

            narration_audio_duration = self._get_audio_duration(narration_audio_path)
            
            probe = ffmpeg.probe(processed_video_path)
            video_info = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            if not video_info:
                raise ValueError("No video stream found in input file")
            video_duration = float(probe['format']['duration'])


      
            self.logger.info(f"Successfully uploaded narration audio and subtitle to COS")
            

            post_process_result = {
                'raw_video_url': merged_video_url,
                'duration': video_duration,
                'narration_audio_url': narration_audio_url,  # Use COS URL instead of local path
                'narration_audio_duration': narration_audio_duration,
                'subtitle_url': subtitle_url,  # Use COS URL instead of local path
                'post_result': metadata_result
            }


            result = self._prepare_result(
                project_no,
                post_process_result,
                clip_desc,
                preset
            )
            
            # 计算总耗时
            steps_timing['total_time'] = int((time.time() - process_start_time))
            
            # 记录处理时间统计
            self.logger.info("Processing time statistics:")
            for step, duration in steps_timing.items():
                self.logger.info(f"  {step}: {duration:.2f} seconds")


            # 验证结果数据并发送回调
            try:
                # 尝试序列化，验证数据是否可以正确转换为JSON
                json_str = json.dumps(result, ensure_ascii=False, indent=2)
                self.logger.info(f'Callback payload (serializable): {json_str}')
                
                # 数据验证通过，发送回调
                if self.callback_url:
                    Thread(target=self.http_util._send_callback, args=(self.callback_url, result)).start()
                
            except (TypeError, ValueError) as e:
                error_msg = f"Result data serialization failed: {str(e)}"
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
                raise Exception(error_msg)
            
            
            return result
            
        except Exception as e:
            error_msg = f"Error in video processing for project {project_no}: {str(e)}"
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

             # 清理slice目录
            slice_project_dir = os.path.join(self.slice_dir, project_no)
            if os.path.exists(slice_project_dir):
                self.logger.info(f"Cleaning slice directory: {slice_project_dir}")
                for item in os.listdir(slice_project_dir):
                    item_path = os.path.join(slice_project_dir, item)
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
            os.makedirs(slice_project_dir, exist_ok=True)

            self.logger.info(f"Successfully cleaned project directories for project: {project_no}")

        except Exception as e:
            self.logger.error(f"Error cleaning project directories: {str(e)}")
            raise Exception(f"Failed to clean project directories: {str(e)}")
        
    
    def _prepare_result(self, project_no, post_process_result, clip_details, preset):
        """
        准备最终返回结果
        
        Args:
            project_no (str): 项目编号
            post_process_result (dict): 后处理结果，包含 raw_video_url, duration, post_result 等
            clip_details (list): 剪辑片段详情
            preset (dict): 预设配置
            
        Returns:
            dict: 标准化的结果数据
        """
        try:
            # 从后处理结果中解构数据
            raw_video_url = post_process_result['raw_video_url']
            duration = post_process_result['duration']
            post_result = post_process_result['post_result']
            narration_audio_url = post_process_result['narration_audio_url']
            subtitle_url = post_process_result['subtitle_url']
            narration_audio_duration = post_process_result['narration_audio_duration']

            def encode_text(text):
                """确保文本使用正确的编码"""
                if not text:
                    return ""
                # 移除引号并进行编码转换
                text = text.strip('"\'')
                try:
                    # 将文本转换为 bytes 后再解码，确保编码正确
                    return text.encode('utf-8', 'ignore').decode('utf-8')
                except UnicodeError:
                    return text
            
            # 构建结果数据，对特定字段进行编码处理
            result = {
                "data": {
                    "projectNo": project_no,
                    "title": encode_text(post_result.get('title')) if post_result.get('title') else None,
                    "description": encode_text(post_result.get('description')) if post_result.get('description') else None,
                    "narration": encode_text(post_result.get('narration')) if preset.get('narration', False) and post_result.get('narration') else "",
                    "videoUrl": raw_video_url,
                    "duration": int(duration) if duration is not None else 0,
                    "clips": clip_details,
                    "narrationAudioUrl": narration_audio_url,
                    "narrationAudioDuration": narration_audio_duration,
                    "subtitleUrl": subtitle_url,
                    "srtUrl": subtitle_url
                }
            }
            
            # 验证结果数据
            try:
                # 先序列化确保数据可以转为JSON
                json_str = json.dumps(result, ensure_ascii=False, indent=2)
                
                # 记录完整的结果数据
                print(
                    "Result data validation successful:\n%s",
                    json_str
                )

            except (TypeError, ValueError) as e:
                self.logger.error(
                    "Result data validation failed for data: %s\nError: %s",
                    str(result)[:1000],  # 限制长度避免日志过大
                    str(e)
                )
                raise Exception(f"Result data serialization failed: {str(e)}")
            
            # 保存结果数据到JSON文件
            self.common_util.save_result_to_json(project_no, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error preparing result data: {str(e)}")
            raise
            
    
    def _verify_ffmpeg_installation(self):
        """验证 FFmpeg 安装和 CUDA 支持"""
        try:
            # 检查 FFmpeg 版本
            version = subprocess.check_output(['ffmpeg', '-version'])
            self.logger.info(f"FFmpeg version: {version.decode()[:200]}...")

            # 检查编码器支持
            encoders = subprocess.check_output(['ffmpeg', '-encoders'])
            if b'h264_nvenc' not in encoders:
                self.logger.warning("NVIDIA GPU encoding (h264_nvenc) not available")
                return False

            # 检查 CUDA 支持
            filters = subprocess.check_output(['ffmpeg', '-filters'])
            if b'scale_cuda' not in filters:
                self.logger.warning("CUDA scaling not available")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error verifying FFmpeg installation: {str(e)}")
            return False