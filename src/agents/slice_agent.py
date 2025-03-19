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

class SliceAgent:
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
            
            # 7. 后处理
            self.logger.info(f"[Step 8/9] Post-processing video")
            start_time = time.time()
            try:
                post_process_result = self.postprocess_agent.post_process_video(
                    synthesized_final_video_url,
                    subtitle_path,
                    narration_audio_path,
                    preset,
                    project_no
                )
                steps_timing['post_process'] = int((time.time() - start_time))
                self.logger.info(f"Post-processing completed in {steps_timing['post_process']:.2f} seconds")
            except Exception as e:
                raise Exception(f"Failed in post-processing: {str(e)}")
            
            # 8. 准备结果
            self.logger.info(f"[Step 9/9] Sending Callback result")
            post_process_result['post_result'] = metadata_result
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