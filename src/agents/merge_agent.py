import os
from src.utils.video_clip_util import VideoClipUtil
from threading import Thread
import logging
from src.utils.tencent_cos_util import COSOperationsUtil
from src.utils.http_util import HttpUtil
from src.utils.gpu_util import GPUUtil
import ffmpeg
from tqdm import tqdm
import subprocess
import time  # 确保在文件顶部导入
import uuid
from urllib.parse import urlparse

class MergeAgent:
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
        self.gpu_util = GPUUtil()
        self.logger = logging.getLogger(__name__)

    '''
    入口函数
    '''
    async def process_merge(self, data):
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
            self.callback_url =  data['callbackUrl']
            narration = data['narration']
            self.logger.info('[step 1]:merge clips with transition effect')
            local_merged_video_path = self._merge_clips_with_transition_effect_by_ffmpeg(
                clips,
                project_no
            )

                        
            upload_response = self.upload_video_to_cos(local_merged_video_path, project_no)
            if upload_response is None:
                raise ValueError("Upload response is None")
            
            merged_video_url = upload_response.get('video_url', '')
            if not merged_video_url:
                raise ValueError("No video URL in upload response")
            
            self.logger.info(f'🎉 Merged video URL: 🔗 {merged_video_url} ✅')
            
            
            self.logger.info('[step 2]:generate narration')
            
            # 生成音频文件
            narration_audio_path = self.video_clip_util._generate_audio(narration, preset, project_no)
            # 3. 生成字幕文件
            subtitle_path = self._generate_subtitles(narration_audio_path, narration)

            '''
            后处理
            '''
            self.logger.info('[step 3]:apply effects')
            processed_video_path, video_duration = self.video_clip_util.apply_effects_with_transition_by_ffmpeg(
                local_merged_video_path, 
                subtitle_path,
                narration_audio_path,
                preset,
                project_no  # 添加 project_no 参数
            )
            
            self.logger.info(f'processed video path: {processed_video_path}')
            self.logger.info(f'video duration: {video_duration}')
            '''
            上传
            '''
            self.logger.info('[step 4]:upload merged video to cos')
            upload_response = self.upload_video_to_cos(processed_video_path, project_no)
            if upload_response is None:
                raise ValueError("Upload response is None")
            
            merged_video_url = upload_response.get('video_url', '')
            if not merged_video_url:
                raise ValueError("No video URL in upload response")

            # Prepare the callback data
            callback_data = {
                "data": {
                    "projectNo": project_no,
                    "videoUrl": merged_video_url,
                    "duration": int(video_duration)  # 向下取整
                }
            }

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
            
            if 'callback_url' in locals():
                Thread(target=self.http_util._send_callback, args=(self.callback_url, error_data)).start()

            return error_data
    
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


    def _merge_clips_with_transition_effect_by_ffmpeg(self, clips, project_no):
        """使用 ffmpeg-python 合并视频并添加转场效果"""
        try:
            # 定义转场效果映射
            xfade_effects = {
                0: "fade",        # Angular -> fade
                1: "fadeblack",   # Bounce -> fadeblack
                2: "fadewhite",   # Burn -> fadewhite
                3: "distance",    # CrossWarp -> distance
                4: "wipeleft",    # Cube -> wipeleft
                5: "wiperight",   # Directional -> wiperight
                6: "fade",        # Fade -> fade
                7: "fadegrays",   # FadeGrayScale -> fadegrays
                8: "smoothleft",  # Morph -> smoothleft
                9: "circleclose", # Rotate -> circleclose
                10: "zoominx",    # SimpleZoom -> zoominx
                11: "squeezev",   # SquaresWire -> squeezev
                12: "circlecrop", # TangentialBlur -> circlecrop
                13: "wipetop"     # Wind -> wipetop
            }

            if not clips:
                raise ValueError("No clips provided")

            # 创建项目目录
            project_dir = os.path.join(self.merge_dir, project_no)
            os.makedirs(project_dir, exist_ok=True)
            self.logger.info(f"Working directory: {project_dir}")

            # 处理视频片段
            video_paths = []
            transition_effects = []
            
            # 1. 处理所有视频和转场效果
            for i, clip in enumerate(clips):
                if clip['type'] == 'video':
                    video_url = clip.get('videoUrl')
                    if not video_url:
                        self.logger.warning(f"Missing videoUrl in clip {i}")
                        continue

                    self.logger.info(f"Processing video {i+1}: {video_url}")
                    
                    local_path = self._download_video_from_cos(video_url, project_no)
                    if not local_path or not os.path.exists(local_path):
                        self.logger.error(f"Failed to download video from {video_url}")
                        continue

                    processed_filename = f"processed_{i}_{os.path.basename(local_path)}"
                    processed_path = os.path.join(project_dir, processed_filename)
                    
                    try:
                        # 获取 GPU 编码参数
                        encoding_params = self.gpu_util.get_encoding_params()
                        self.logger.info(f"Using encoding params for video processing: {encoding_params}")
                        
                        stream = (
                            ffmpeg
                            .input(local_path)
                            .filter('scale', 1080, 1920, force_original_aspect_ratio='decrease')
                            .filter('pad', 1080, 1920, '(ow-iw)/2', '(oh-ih)/2')
                            .output(processed_path, **encoding_params)
                            .overwrite_output()
                        )
                        
                        self.logger.info(f"Processing video with ffmpeg-python using GPU: {processed_path}")
                        stream.run(capture_stderr=True)
                        
                        video_paths.append(processed_path)
                        self.logger.info(f"Successfully processed video to: {processed_path}")
                        
                        os.remove(local_path)  # 清理原始下载文件
                        
                    except ffmpeg.Error as e:
                        self.logger.error(f"FFmpeg error: {e.stderr.decode() if e.stderr else 'No error output'}")
                        # 如果 GPU 处理失败，回退到 CPU 处理
                        self.logger.info("Falling back to CPU processing...")
                        stream = (
                            ffmpeg
                            .input(local_path)
                            .filter('scale', 1080, 1920, force_original_aspect_ratio='decrease')
                            .filter('pad', 1080, 1920, '(ow-iw)/2', '(oh-ih)/2')
                            .output(processed_path,
                                vcodec='libx264',
                                acodec='aac',
                                preset='medium',
                                pix_fmt='yuv420p',
                                crf=23)
                            .overwrite_output()
                        )
                        stream.run(capture_stderr=True)
                        continue
                        
                elif clip['type'] == 'effect':
                    effect = xfade_effects.get(clip.get('transitionEffect', 0), 'fade')
                    transition_effects.append(effect)
                    self.logger.info(f"Added transition effect: {effect}")

            # 2. 检查处理后的视频数量
            if len(video_paths) < 2:
                if len(video_paths) == 1:
                    self.logger.info("Only one video processed, returning it directly")
                    return video_paths[0]
                raise ValueError(f"Not enough valid videos to merge (found {len(video_paths)})")

            try:
                # 3. 合并视频并添加转场效果
                output_path = os.path.join(project_dir, f"{project_no}_final_{uuid.uuid4()}.mp4")
                current_input = video_paths[0]
                
                for i in range(1, len(video_paths)):
                    temp_output = f"{output_path}.temp{i}.mp4"
                    effect = transition_effects[i-1] if i-1 < len(transition_effects) else 'fade'
                    
                    try:
                        # 获取第一个视频的持续时间
                        probe = ffmpeg.probe(current_input)
                        duration = float(probe['streams'][0]['duration'])
                        
                        self.logger.info(f"[🎬 ffmpeg] Applying transition ✨ {effect} ✨ between videos {i} and {i+1} 🔄")
                        # 使用 ffmpeg-python 构建转场效果
                        input1 = ffmpeg.input(current_input)
                        input2 = ffmpeg.input(video_paths[i])
                        
                        joined = ffmpeg.filter(
                            [input1, input2],
                            'xfade',
                            transition=effect,
                            duration=1,
                            offset=duration-1
                        )
                        
                       # 使用 GPU 工具类获取编码参数
                        output_options = self.gpu_util.get_encoding_params()
                        self.logger.info(f"Using encoding options: {output_options}")
                        
                        stream = ffmpeg.output(
                            joined,
                            temp_output,
                            **output_options
                        ).overwrite_output()
                        
                        stream.run(capture_stderr=True)
                        
                        if not os.path.exists(temp_output):
                            raise ValueError(f"Failed to create transition output: {temp_output}")
                        
                        current_input = temp_output
                        
                    except ffmpeg.Error as e:
                        self.logger.error(f"FFmpeg error during transition: {e.stderr.decode() if e.stderr else 'No error output'}")
                        # 如果转场失败，尝试直接连接
                        self.logger.info("Falling back to direct concatenation...")
                        
                        try:
                            stream = (
                                ffmpeg
                                .concat(
                                    ffmpeg.input(current_input),
                                    ffmpeg.input(video_paths[i])
                                )
                                .output(temp_output)
                                .overwrite_output()
                            )
                            stream.run(capture_stderr=True)
                            current_input = temp_output
                            
                        except ffmpeg.Error as concat_error:
                            self.logger.error(f"Concat error: {concat_error.stderr.decode() if concat_error.stderr else 'No error output'}")
                            raise

                # 4. 重命名最后的输出文件
                os.rename(current_input, output_path)
                self.logger.info(f"Successfully created merged video: {output_path}")
                return output_path

            finally:
                # 清理临时文件
                for video_path in video_paths:
                    try:
                        if os.path.exists(video_path):
                            os.remove(video_path)
                    except Exception as e:
                        self.logger.warning(f"Failed to cleanup file {video_path}: {str(e)}")
                
                # 清理临时转场文件
                for i in range(1, len(video_paths)):
                    temp_file = f"{output_path}.temp{i}.mp4"
                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    except Exception as e:
                        self.logger.warning(f"Failed to cleanup temp file {temp_file}: {str(e)}")

        except Exception as e:
            self.logger.error(f"Failed to merge clips with transitions: {str(e)}")
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
        
        