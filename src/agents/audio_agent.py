import logging
from src.utils.video_clip_util import VideoClipUtil
import subprocess
import os
import time

class AudioAgent:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.callback_url = None
        self.video_clip_util = VideoClipUtil()
        self.project_dir = os.path.join(os.getcwd(), 'temp')
  
    def sync_audio_with_video(self, selected_clips, preset, project_no):
        """同步音频与视频，确保音频和视频时长匹配"""        
        try:
            # 生成标题和描述
            metadata_result = self.video_clip_util.generate_video_metadata(
                selected_clips,
                preset
            )
            
            # 生成旁白
            narration_content = self.video_clip_util.generate_narration_directly(
                selected_clips, 
                preset
            )
            
            # 生成音频文件
            narration_audio_path = self.video_clip_util._generate_audio(narration_content, preset, project_no)
            
            # 同步视频片段并生成字幕
            sync_selected_clips, subtitle_path = self._sync_clips_with_audio(
                selected_clips, 
                narration_audio_path,
                narration_content,
                preset
            )
            metadata_result['narration'] = narration_content

            return narration_audio_path, sync_selected_clips, metadata_result, subtitle_path
            
        except Exception as e:
            self.logger.error(f"Error in _sync_audio_with_video: {str(e)}")
            raise
    

    def _sync_clips_with_audio(self, selected_clips,narration_audio_path,narration_content,preset):
        """将视频片段与音频时长同步，优先使用LLM方案，失败时降级到传统方案"""
        total_video_duration = sum(clip['end'] - clip['start'] for clip in selected_clips)
        self.logger.info(f"Original video duration: {total_video_duration} seconds")
        
        try:
            # 首先尝试使用LLM方案
            self.logger.info("Attempting to sync clips using LLM approach...")
            # 获取音频时长
            audio_duration = self._get_audio_duration(narration_audio_path)
            self.logger.info(f"Generated audio duration: {audio_duration} seconds")
            
            synced_clips = self.video_clip_util.sync_clips_with_audio(selected_clips, audio_duration,preset)
             
            # 3. 生成字幕文件
            subtitle_path = self._generate_subtitles(narration_audio_path, narration_content,preset)
            
            # 验证LLM结果
            if synced_clips:
                return synced_clips, subtitle_path
            else:
                self.logger.warning("LLM returned empty clips, falling back to traditional method")
 
        except Exception as e:
            self.logger.error(f"LLM sync failed: {str(e)}, falling back to traditional method")
            raise


             
    def _generate_subtitles(self, narration_audio_path, narration_content,preset):
        """生成 SRT 格式字幕文件"""
        try:
            subtitle_path = os.path.join(self.project_dir, f"subtitle_{int(time.time())}.srt")
            
            # 首先尝试使用 LLM 获取音频分段
            try:
                audio_segments = self.video_clip_util._get_audio_segments(narration_audio_path,preset)
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
    



