import logging
import requests
import json
from threading import Thread
import os
import hashlib
from urllib.parse import urlparse
from src.utils.tencent_cos_util import COSOperationsUtil

class HttpUtil:
    def __init__(self):
        self.cos_ops_util = COSOperationsUtil()
        self.logger = logging.getLogger(__name__)
        self.callback_url = None

    def _send_status_callback(self, callback_url, status_data):
        """发送状态回调"""
        if not callback_url:
            return
        
        try:
            Thread(target=self._send_callback, args=(callback_url, status_data)).start()
        except Exception as e:
            self.logger.error(f"Failed to send status callback: {str(e)}")

    
    def _send_callback(self, callback_url, data):
        """
        发送回调请求到指定URL
        
        Args:
            callback_url (str): 回调URL
            data (dict): 要发送的数据
            
        Returns:
            bool: 发送是否成功
        """
        try:
            # 递归处理数据，确保所有字符串都使用 utf8mb4 编码
            def encode_utf8mb4(obj):
                if isinstance(obj, dict):
                    return {k: encode_utf8mb4(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [encode_utf8mb4(item) for item in obj]
                elif isinstance(obj, str):
                    # 将字符串转换为 utf8mb4 编码
                    try:
                        # 先编码为 bytes，再解码为 utf8mb4
                        return obj.encode('utf-8').decode('utf-8')
                    except UnicodeError:
                        # 如果编码失败，使用 replace 策略
                        return obj.encode('utf-8', 'replace').decode('utf-8')
                return obj

            # 处理数据，确保所有字符串都是 utf8mb4 编码
            encoded_data = encode_utf8mb4(data)
            
            # 序列化为 JSON
            json_data = json.dumps(
                encoded_data,
                ensure_ascii=False,
                indent=2
            )
            
            self.logger.info(f'===> Send callback data: {json_data}')
            
            headers = {
                'Content-Type': 'application/json; charset=utf8mb4'
            }
            
            # 使用 utf8mb4 编码发送数据
            response = requests.post(
                callback_url,
                data=json_data.encode('utf-8'),
                headers=headers,
                timeout=300
            )
            
            if response.status_code == 200:
                response_json = response.json()
                self.logger.info(f'Callback response: {json.dumps(response_json, ensure_ascii=False, indent=2)}')
                return True
            else:
                error_message = response.json().get('error', {}).get('message', 'Unknown error')
                self.logger.error(
                    f"Failed to send callback to {callback_url}. "
                    f"Status code: {response.status_code}. "
                    f"Error: {error_message}"
                )

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error when sending callback to {callback_url}: {str(e)}")
        except ValueError as e:
            self.logger.error(f"Invalid JSON response from {callback_url}: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error when sending callback to {callback_url}: {str(e)}")

    def _download_video_from_cos(self, cos_video_url, project_no, filename_prefix=""):
        """
        从COS下载视频，添加唯一性标识
        
        Args:
            cos_video_url (str): COS视频URL或路径
            project_no (str): 项目编号
            filename_prefix (str): 文件名前缀，用于标识唯一性
            
        Returns:
            str: 本地文件路径
        """
        video_local_path = None
        try:
            print(f'cos video url: {cos_video_url}')

                        # 解析URL获取远程路径
            if 'https' in cos_video_url:
                parsed_url = urlparse(cos_video_url)
                remote_path = parsed_url.path.lstrip('/')

                bucket_name = getattr(self.cos_ops_util, 'bucket_name', None)
                if bucket_name and remote_path.startswith(bucket_name):
                    remote_path = remote_path[len(bucket_name):].lstrip('/')
            else:
                remote_path = cos_video_url

            # 生成唯一文件名
            file_extension = os.path.splitext(remote_path)[1]
            url_hash = hashlib.md5(cos_video_url.encode()).hexdigest()[:8]
            unique_filename = f"{filename_prefix}{url_hash}{file_extension}"
            
            # 创建临时目录路径
            temp_dir = os.path.join(os.getcwd(), 'temp', project_no)
            video_local_path = os.path.join(temp_dir, unique_filename)

            # 确保目录存在
            os.makedirs(temp_dir, exist_ok=True)

            # 下载文件
            print(f"\033[93mDownloading {remote_path} to {video_local_path}\033[0m")
            try:
                # 如果文件已存在且大小不为0，跳过下载
                if os.path.exists(video_local_path) and os.path.getsize(video_local_path) > 0:
                    print(f"File already exists and is valid: {video_local_path}")
                    return video_local_path

                # 使用COS工具下载文件
                self.cos_ops_util.download_file(
                    remote_path,
                    video_local_path
                )
            except Exception as e:
                self.logger.error(f"Failed to download from COS: {str(e)}")
                raise
                
            # 验证下载结果
            if not os.path.exists(video_local_path):
                raise FileNotFoundError(f"File download failed - file not found at {video_local_path}")
            
            if os.path.getsize(video_local_path) == 0:
                os.remove(video_local_path)  # 删除空文件
                raise ValueError(f"Downloaded file is empty: {video_local_path}")

            print(f"Successfully downloaded video to {video_local_path}")
            return video_local_path
                
        except Exception as e:
            self.logger.error(f"\033[93mError downloading video from {cos_video_url}: {str(e)}\033[0m")
            # 检查 video_local_path 是否为有效路径
            if video_local_path and isinstance(video_local_path, str):
                try:
                    if os.path.exists(video_local_path):
                        os.remove(video_local_path)  # 清理失败的下载
                        self.logger.info(f"Cleaned up failed download: {video_local_path}")
                except Exception as cleanup_error:
                    self.logger.warning(f"Failed to clean up file: {cleanup_error}")
            return None
        
    def download_video_from_cos(self, cos_video_url, project_no, filename_prefix=""):
        """
        从COS下载视频，添加唯一性标识
        
        Args:
            cos_video_url (str): COS视频URL或路径
            project_no (str): 项目编号
            filename_prefix (str): 文件名前缀，用于标识唯一性
            
        Returns:
            str: 本地文件路径
        """
        video_local_path = None
        try:
            print(f'cos video url: {cos_video_url}')

                        # 解析URL获取远程路径
            if 'https' in cos_video_url:
                parsed_url = urlparse(cos_video_url)
                remote_path = parsed_url.path.lstrip('/')

                bucket_name = getattr(self.cos_ops_util, 'bucket_name', None)
                if bucket_name and remote_path.startswith(bucket_name):
                    remote_path = remote_path[len(bucket_name):].lstrip('/')
            else:
                remote_path = cos_video_url

            # 生成唯一文件名
            file_extension = os.path.splitext(remote_path)[1]
            url_hash = hashlib.md5(cos_video_url.encode()).hexdigest()[:8]
            unique_filename = f"{filename_prefix}{url_hash}{file_extension}"
            
            # 创建临时目录路径
            temp_dir = os.path.join(os.getcwd(), 'temp', project_no)
            video_local_path = os.path.join(temp_dir, unique_filename)

            # 确保目录存在
            os.makedirs(temp_dir, exist_ok=True)

            # 下载文件
            print(f"\033[93mDownloading {remote_path} to {video_local_path}\033[0m")
            try:
                # 如果文件已存在且大小不为0，跳过下载
                if os.path.exists(video_local_path) and os.path.getsize(video_local_path) > 0:
                    print(f"File already exists and is valid: {video_local_path}")
                    return video_local_path

                # 使用COS工具下载文件
                self.cos_ops_util.download_file(
                    remote_path,
                    video_local_path
                )
            except Exception as e:
                self.logger.error(f"Failed to download from COS: {str(e)}")
                raise
                
            # 验证下载结果
            if not os.path.exists(video_local_path):
                raise FileNotFoundError(f"File download failed - file not found at {video_local_path}")
            
            if os.path.getsize(video_local_path) == 0:
                os.remove(video_local_path)  # 删除空文件
                raise ValueError(f"Downloaded file is empty: {video_local_path}")

            print(f"Successfully downloaded video to {video_local_path}")
            return video_local_path
                
        except Exception as e:
            self.logger.error(f"\033[93mError downloading video from {cos_video_url}: {str(e)}\033[0m")
            # 检查 video_local_path 是否为有效路径
            if video_local_path and isinstance(video_local_path, str):
                try:
                    if os.path.exists(video_local_path):
                        os.remove(video_local_path)  # 清理失败的下载
                        self.logger.info(f"Cleaned up failed download: {video_local_path}")
                except Exception as cleanup_error:
                    self.logger.warning(f"Failed to clean up file: {cleanup_error}")
            return None