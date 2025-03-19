import logging
from src.utils.video_clip_util import VideoClipUtil
from src.utils.tencent_cos_util import COSOperationsUtil
from src.utils.http_util import HttpUtil
import os
import uuid
from threading import Thread
import ffmpeg
from urllib.parse import urlparse

class SynthesizeAgent:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.callback_url = None
        self.video_clip_util = VideoClipUtil()
        self.cos_ops_util = COSOperationsUtil()
        self.http_util = HttpUtil()
        self.project_dir = os.path.join(os.getcwd(), 'temp')
        self.base_cos_url = os.getenv('BASE_COS_URL')

    def synthesize_final_video(self, selected_clips, merged_input_video_url,project_no):
        """合成最终视频"""
        try:
            selected_clips_directory = [{
                'video_url': merged_input_video_url,
                'segments': selected_clips
            }]
            
            merged_video_url, clip_details = self.synthesize_video_by_local(
                selected_clips_directory,
                project_no
            )
            
            return merged_video_url, clip_details
            
        except Exception as e:
            self.logger.error(f"Failed to synthesize final video: {str(e)}")
            if self.callback_url:
                Thread(target=self.http_util._send_callback, args=(self.callback_url, {
                        "data":{
                            "projectNo": project_no,
                        },
                        "error": {
                            "code": 500,
                            "message": "Failed to synthesize final video"
                        }})).start()
            raise


    

    def synthesize_video_by_local(self, selected_clips_directory, project_no):
        temp_files = []  # Track files to clean up
        try:
            merged_video_path, clip_infos = self.synthesize_video(
                selected_clips_directory=selected_clips_directory,
                project_no=project_no
            )

            if merged_video_path:
                temp_files.append(merged_video_path)
            
            # Add clip paths to temp_files for cleanup
            clip_paths = [info['path'] for info in clip_infos]
            temp_files.extend(clip_paths)

            # Upload individual clips to COS
            clip_urls = self.upload_clips_to_cos(
                clip_paths=clip_paths,
                project_no=project_no
            )
            
            # Upload merged video to COS
            upload_response = self.upload_video_to_cos(
                local_video_path=merged_video_path,
                project_no=project_no
            )

            # 检查 upload_response 是否为 None
            if upload_response is None:
                raise ValueError("Upload response is None")
                
            merged_video_url = upload_response.get('video_url', '')
            if not merged_video_url:
                raise ValueError("No video URL in upload response")

            # Combine clip URLs with their durations
            clip_details = []
            for i, clip_info in enumerate(clip_infos):
                clip_details.append({
                    'url': clip_urls[i]['url'],
                    'duration': clip_info['duration']
                })

            # 格式化打印日志
            print("\n=== Video Processing Results ===")
            print("Clips:")
            for i, clip in enumerate(clip_details, 1):
                print(f"  Clip {i}:")
                print(f"    URL: {clip['url']}")
                print(f"    Duration: {clip['duration']} seconds")
            print("\nMerged Video:")
            print(f"  URL: {merged_video_url}")
            print("============================\n")

            # 返回与 remote 相同的格式
            return (
                merged_video_url,  # merged_video_url
                clip_details      # clip_urls
            )
            
        except Exception as e:
            print(f'Error: {str(e)}')
            raise ValueError(f"Video synthesis failed: {str(e)}")

        finally:
            # Clean up temporary files after successful upload
            for temp_file in temp_files:
                try:
                    if temp_file and os.path.exists(temp_file):
                        os.remove(temp_file)
                except Exception as cleanup_error:
                    print(f"Failed to clean up file {temp_file}: {str(cleanup_error)}")


    '''
    上传切片
    '''
    def upload_clips_to_cos(self, clip_paths, project_no):
        items = []
        failed_uploads = []
        
        for clip_path in clip_paths:
            try:
                # Generate a UUID for this upload
                upload_uuid = str(uuid.uuid4())
                
                # Create the COS path with project number, UUID, and 'clips' folder
                filename = os.path.basename(clip_path)
                remote_cos_clip_path = f"{project_no}/clips/{upload_uuid}_{filename}"
                
                # Upload the clip to COS
                self.cos_ops_util.upload_file(
                    local_file_path=clip_path,
                    cos_file_path=remote_cos_clip_path
                )
                
                item = {
                    "url": f"{self.base_cos_url}/{remote_cos_clip_path}",
                }
                items.append(item)
                
                self.logger.info(f"Uploaded clip to COS: {remote_cos_clip_path}")
            except Exception as e:
                self.logger.error(f"Error uploading clip {clip_path} to COS: {str(e)}")
                failed_uploads.append({"path": clip_path, "error": str(e)})
                continue
        
        if failed_uploads:
            self.logger.error(f"Failed to upload {len(failed_uploads)} clips: {failed_uploads}")
            if not items:
                raise Exception(f"All uploads failed: {failed_uploads}")
        
        return items
    

    '''
    上传合成视频
    '''
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
            
            # Generate and return the public URL
            video_url = f"{self.base_cos_url}/{remote_cos_path}"
            self.logger.info(f"Successfully uploaded merged video to COS: {remote_cos_path}")
            
            return {
                "video_url": video_url,
                "project_no": project_no
            }
            
        except Exception as e:
            self.logger.error(f"Error uploading merged video {local_video_path} to COS: {str(e)}")
            return None
        

    
    
    def synthesize_video(self, selected_clips_directory, project_no):
        """
        Synthesize video clips based on selected segments and merge them.
        
        Args:
            materials (list): List of video materials with URLs
            selected_clips_directory (list): List of dictionaries containing video URLs and their selected segments
                Format: [{'video_url': str, 'segments': list[dict]}]
            project_no (str): Project identifier
            
        Returns:
            tuple: (merged_video_path, all_clip_paths, temp_files)
                - merged_video_path: Path to the final merged video
                - all_clip_paths: List of paths to individual clips
                - temp_files: List of temporary files created
                
        Raises:
            ValueError: If no valid clips could be generated or merged
        """
        clip_info_list = []

        try:
            # Generate video clips based on selected segments
            '''
            可以接受的clip的片段，等待合成
            '''
            if selected_clips_directory:
                for video_clips in selected_clips_directory:
                    video_url = video_clips['video_url']
                    segments = video_clips['segments']
                    
                    if not segments:
                        self.logger.warning(f"No segments selected for video: {video_url}")
                        continue
                        
                    '''
                    download origin videos
                    '''
                    video_local_path = self.download_video_from_cos(video_url, project_no)
                    self.logger.info(f"local video path: {video_local_path}")
                    if not video_local_path:
                        raise ValueError(f"Failed to download video from {video_url}")
                    
                    self.logger.info(f"Generating clips from video: {video_url}")
                    self.logger.info(f"Using clip segments: {segments}")
                    
                    # Generate clips using the selected segments
                    clip_paths = self.video_clip_util.generate_video_clips_by_local(
                        project_no,
                        segments,  # Format expected by generate_video_clips
                        video_local_path
                    )
                    
                    if clip_paths:
                        # 获取每个片段的时长并记录
                        for clip_path in clip_paths:
                            try:
                                # 使用ffmpeg获取视频时长
                                probe = ffmpeg.probe(clip_path)
                                duration = float(probe['streams'][0]['duration'])  # 使用 float 而不是 int
                                rounded_duration = round(duration)  # 四舍五入到最近的整数
                                
                                clip_info = {
                                    'path': clip_path,
                                    'duration': rounded_duration
                                }
                                clip_info_list.append(clip_info)
                            except Exception as e:
                                self.logger.error(f"Error getting duration for clip {clip_path}: {str(e)}")
                                # 如果获取时长失败，仍然添加路径但时长为None
                                clip_info_list.append({
                                    'path': clip_path,
                                    'duration': None
                                })
                        self.logger.info(f"Generated {len(clip_paths)} clips")
                    else:
                        self.logger.warning(f"No clips generated for video: {video_url}")

            if not clip_info_list:
                raise ValueError("No valid clips were generated from any video")

            # 获取所有clip路径用于合并
            all_clip_paths = [info['path'] for info in clip_info_list]

            # Merge the generated clips
            raw_merged_filename = f"{project_no}_raw_merged.mp4"
            raw_merged_video_path = os.path.join(self.project_dir, project_no, raw_merged_filename)
            
            raw_merged_video_path = self.video_clip_util.merge_video_clips(all_clip_paths, raw_merged_video_path)
            if not raw_merged_video_path:
                raise ValueError("Failed to merge video clips")

            return raw_merged_video_path, clip_info_list

        except Exception as e:
            self.logger.error(f"Error synthesizing video: {str(e)}")
            raise  # Re-raise the exception instead of returning None



    
    '''
    下载视频
    '''
    def download_video_from_cos(self, cos_video_url, project_no):
        """Download video from COS to local storage"""
        try:
            parsed_url = urlparse(cos_video_url)
            if not parsed_url.path:
                raise ValueError("Invalid COS URL: no path found: {parsed_url}")

            object_key = parsed_url.path.lstrip('/')
            original_filename = os.path.basename(object_key)
            
            if not original_filename:
                raise ValueError("Invalid COS URL: no filename found")
            
            video_local_path = os.path.join(os.getcwd(), 'temp', project_no, original_filename)

            # Ensure the directory exists
            os.makedirs(os.path.dirname(video_local_path), exist_ok=True)

            # Download the file
            self.cos_ops_util.download_file(
                object_key,
                video_local_path
            )
            
            if not os.path.exists(video_local_path):
                raise FileNotFoundError("File download failed")

            return video_local_path
            
        except Exception as e:
            print(f"Error downloading video: {str(e)}")
            return None
        
