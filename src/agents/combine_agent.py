from src.utils.video_clip_util import VideoClipUtil
from src.utils.tencent_cos_util import COSOperationsUtil
from src.utils.http_util import HttpUtil
import logging
import os
import time
import requests
from threading import Thread
import hashlib
from tqdm import tqdm
import subprocess
import uuid
import json

class CombineAgent:
    def __init__(self):
        self.cos_ops_util = COSOperationsUtil()
        self.logger = logging.getLogger(__name__)
        self.callback_url = None
        self.http_util = HttpUtil()

    def combine_raw_videos(self, materials, project_no):
        """合并输入视频并上传到COS"""
        local_merged_video_path = None  # 初始化变量
        processed_files = []  # 跟踪所有处理过的文件
        
        try:
            video_urls = [m['url'] for m in materials if m['type'] == 'video']
            self.logger.info(f"[Step 1/8] Merging {len(video_urls)} input videos")
            
            # 1. 合并视频
            self.logger.info(f'===>开始合成视频，包括: {video_urls}')
            local_merged_video_path = self.combine_input_videos(video_urls, project_no)
            if not local_merged_video_path or not os.path.exists(local_merged_video_path):
                raise ValueError("===> 合成失败: Failed to merge videos: no valid output file")
            
            self.logger.info(f'===> 合成成功: {local_merged_video_path}')
            # 2. 验证本地文件
            if os.path.getsize(local_merged_video_path) == 0:
                raise ValueError("Merged video file is empty")
            
            # 3. 上传到 COS
            try:
                filename = os.path.basename(local_merged_video_path)
                cos_key = f"merged_videos/{project_no}/{filename}"
                
                # 上传文件并等待完成
                upload_success = self.cos_ops_util.upload_file(
                    local_file_path=local_merged_video_path,
                    cos_file_path=cos_key,
                )
                
                if not upload_success:
                    raise Exception("Upload to COS failed")
                
                # 4. 获取并验证 COS URL
                max_retries = 3
                cos_url = None
                
                for attempt in range(max_retries):
                    time.sleep(3)  # 等待文件同步
                    cos_url = self.cos_ops_util.get_file_url(cos_key)
                    if cos_url:
                        # 验证URL可访问性
                        try:
                            response = requests.head(cos_url, timeout=10)
                            if response.status_code == 200:
                                self.logger.info(f"\033[92mSuccessfully uploaded and verified merged video at: {cos_url}\033[0m")
                                return cos_url
                        except requests.RequestException:
                            self.logger.info(f"\033[93mAttempt {attempt + 1}: URL not yet accessible\033[0m")
                            continue
                
                raise Exception(f"Failed to verify COS URL after {max_retries} attempts")
                
            except Exception as e:
                error_msg = f"Failed to upload or verify merged video in COS: {str(e)}"
                self.logger.error(f"\033[91m{error_msg}\033[0m")
                raise Exception(error_msg)
            
        except Exception as e:
            error_msg = f"===> 合成失败: Failed in video merge and upload process: {str(e)}"
            self.logger.error(f"\033[91m{error_msg}\033[0m")
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
            
        finally:
            # 清理所有临时文件
            try:
                # 清理合并后的视频文件
                if local_merged_video_path and os.path.exists(local_merged_video_path):
                    os.remove(local_merged_video_path)
                    self.logger.info(f"\033[93mCleaned up merged video file\033[0m")
                
                # 清理处理过的临时文件
                for temp_file in processed_files:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        self.logger.info(f"\033[93mCleaned up temporary file: {temp_file}\033[0m")
                        
            except Exception as cleanup_error:
                self.logger.warning(f"Failed to clean up some temporary files: {str(cleanup_error)}")





    def combine_input_videos(self, video_urls, project_no):
        """快速合并输入视频"""
        try:
            if not video_urls:
                self.logger.error("Empty video URL list provided")
                return None
                
            # 1. 下载视频
            unique_urls = list(dict.fromkeys(video_urls))
            media_paths = []
            
            for url in tqdm(unique_urls, desc="Downloading videos"):
                try:
                    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                    local_path = self.http_util._download_video_from_cos(
                        url, 
                        project_no,
                        filename_prefix=f"unique_{url_hash}_"
                    )
                    
                    if local_path:
                        media_paths.append(local_path)
                    else:
                        self.logger.error(f"Failed to download video: {url}")
                        
                except Exception as e:
                    self.logger.error(f"Error downloading {url}: {str(e)}")
                    continue
            
            if not media_paths:
                raise ValueError("No videos were successfully downloaded")

            try:
                # 2. 创建合并列表文件
                concat_file = os.path.join(
                    os.path.dirname(media_paths[0]), 
                    f'concat_{uuid.uuid4()}.txt'
                )
                
                # 写入文件列表
                with open(concat_file, 'w', encoding='utf-8') as f:
                    for path in media_paths:
                        f.write(f"file '{path}'\n")

                # 3. 设置输出路径
                output_filename = f"merged_{project_no}.mp4"
                local_merged_video_path = os.path.join(
                    os.path.dirname(media_paths[0]), 
                    output_filename
                )

                # 4. 尝试直接拼接（不使用GPU，因为只是流复制）
                fast_cmd = [
                    'ffmpeg',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', concat_file,
                    '-c', 'copy',              # 直接复制流
                    '-movflags', '+faststart',
                    '-y',
                    local_merged_video_path
                ]
                
                self.logger.info("Attempting fast merge...")
                process = subprocess.Popen(
                    fast_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                stdout, stderr = process.communicate()
                
                # 如果直接拼接失败，尝试使用GPU转码并合并
                if process.returncode != 0:
                    error_msg = stderr.decode('utf-8')
                    self.logger.error(f"Stream copy merge failed: {error_msg}")
                    
                    # 尝试使用更严格的参数进行流复制
                    self.logger.info("Trying alternative stream copy approach...")
                    alt_cmd = [
                        'ffmpeg',
                        '-f', 'concat',
                        '-safe', '0',
                        '-i', concat_file,
                        '-c', 'copy',              # 仍然使用流复制
                        '-strict', '-2',           # 更宽松的兼容性设置
                        '-max_muxing_queue_size', '1024',  # 增加复用队列大小
                        '-y',
                        local_merged_video_path
                    ]
                    
                    process = subprocess.Popen(
                        alt_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    
                    stdout, stderr = process.communicate()
                    
                    if process.returncode != 0:
                        error_msg = stderr.decode('utf-8')
                        raise Exception(f"GPU merge failed: {error_msg}")
                
                # 验证输出文件
                if os.path.exists(local_merged_video_path) and os.path.getsize(local_merged_video_path) > 0:
                    self.logger.info(f"Successfully merged videos to: {local_merged_video_path}")
                    return local_merged_video_path
                else:
                    raise ValueError("Merged video file is missing or empty")
                    
            except Exception as e:
                self.logger.error(f"Error in merge process: {str(e)}")
                raise
                
            finally:
                # 清理临时文件
                try:
                    if os.path.exists(concat_file):
                        os.remove(concat_file)
                    for path in media_paths:
                        if os.path.exists(path):
                            os.remove(path)
                except Exception as e:
                    self.logger.warning(f"Error cleaning up temporary files: {str(e)}")
                
        except Exception as e:
            self.logger.error(f"Failed to merge input videos: {str(e)}")
            return None

