import logging
import time
from threading import Thread
from tqdm import tqdm
import ffmpeg
import requests
import os
from src.utils.http_util import HttpUtil
from src.utils.common_util import CommonUtil


class FrameAgent:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.callback_url = None
        self.interval_seconds = int(os.getenv('INTERVAL_SECONDS'))
        self.http_util = HttpUtil()
        self.common_util = CommonUtil()

    def _get_video_duration(self, video_url):
        """获取视频时长"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 首先尝试使用 ffprobe
                probe = ffmpeg.probe(video_url, cmd='ffprobe')
                for stream in probe['streams']:
                    if stream['codec_type'] == 'video':
                        duration = float(stream.get('duration', 0))
                        if duration > 0:
                            self.logger.info(f"Video duration from ffprobe: {duration} seconds")
                            return duration
                
                # 如果在视频流中没找到时长，尝试从格式信息中获取
                if 'format' in probe and 'duration' in probe['format']:
                    duration = float(probe['format']['duration'])
                    self.logger.info(f"Video duration from format info: {duration} seconds")
                    return duration
                
            except ffmpeg.Error as e:
                self.logger.warning(f"FFprobe attempt {attempt + 1} failed: {str(e)}")
                if e.stderr:
                    self.logger.warning(f"FFprobe stderr: {e.stderr.decode('utf-8')}")
                
                # 如果是最后一次尝试，尝试其他方法
                if attempt == max_retries - 1:
                    try:
                        # 尝试使用 HTTP HEAD 请求
                        response = requests.head(video_url, timeout=10)
                        if 'Content-Duration' in response.headers:
                            duration = float(response.headers['Content-Duration'])
                            self.logger.info(f"Video duration from HTTP headers: {duration} seconds")
                            return duration
                        
                        # 如果有 Content-Length，尝试下载一小部分来获取时长
                        if 'Content-Length' in response.headers:
                            try:
                                # 只下载视频的前几秒来获取时长
                                probe = ffmpeg.probe(
                                    video_url, 
                                    cmd='ffprobe',
                                    extra_args=['-read_ahead_limit', '10000000']  # 限制读取大小
                                )
                                if 'format' in probe and 'duration' in probe['format']:
                                    duration = float(probe['format']['duration'])
                                    self.logger.info(f"Video duration from partial probe: {duration} seconds")
                                    return duration
                            except Exception as probe_err:
                                self.logger.warning(f"Partial probe failed: {str(probe_err)}")
                    
                    except requests.RequestException as req_err:
                        self.logger.warning(f"HTTP request failed: {str(req_err)}")
            
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed with unexpected error: {str(e)}")
                
            # 在重试之前等待一小段时间
            if attempt < max_retries - 1:
                time.sleep(1)
        
        # 如果所有方法都失败了，记录详细错误并抛出异常
        error_msg = "Failed to get video duration after all attempts"
        self.logger.error(error_msg)
        raise Exception(error_msg)
    

        
    def process_segments(self, captions, video_url, project_no):
        """处理视频片段"""
        try:
            segments = []
            
            if not captions or not captions.get('success'):
                raise Exception("No valid captions generated")
            
            # 获取视频时长，添加重试逻辑
            max_retries = 3
            video_duration = None
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    video_duration = self._get_video_duration(video_url)
                    if video_duration:
                        break
                except Exception as e:
                    last_error = e
                    self.logger.warning(f"Attempt {attempt + 1} to get video duration failed: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(1)
            
            if not video_duration:
                raise Exception(f"Failed to get video duration after {max_retries} attempts: {str(last_error)}")
            
            # 处理每一帧
            for frame in tqdm(captions['frames'], desc="Processing frames"):
                start_time = frame.get('timestamp', 0)
                caption = frame.get('caption', '')
                end_time = start_time + self.interval_seconds
                if end_time > video_duration:
                    end_time = video_duration
                    
                segments.append({
                    'start': start_time,
                    'end': end_time,
                    'text': caption
                })
                
            # 保存分段信息
            self.common_util.save_segments_to_json(project_no, segments, video_url)
            return segments
            
        except Exception as e:
            self.logger.error(f"Failed to process segments: {str(e)}")
            if self.callback_url:
                Thread(target=self.http_util._send_callback, args=(self.callback_url, {
                        "data":{
                            "projectNo": project_no,
                        },
                        "error": {
                            "code": 500,
                            "message": "Failed to process segments"
                        }})).start()
            raise