import logging
from threading import Thread
from src.utils.http_util import HttpUtil
from src.utils.video_clip_util import VideoClipUtil
import time

class ClipAgent:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.callback_url = None
        self.http_util = HttpUtil()
        self.video_clip_util = VideoClipUtil()

    def apply_clip_strategy(self, merged_segments, preset, project_no):
        """应用剪辑策略"""
        try:
            selected_clips = self.process_clip_strategy(
                segments=merged_segments,
                preset=preset,
                project_no=project_no
            )
            
            if not selected_clips:
                raise Exception("No clips were selected after processing")
                
            return selected_clips
            
        except Exception as e:
            self.logger.error(f"Failed to apply clip strategy: {str(e)}")
            if self.callback_url:
                Thread(target=self.http_util._send_callback, args=(self.callback_url, {
                        "data":{
                            "projectNo": project_no,
                        },
                        "error": {
                            "code": 500,
                            "message": "Failed to apply clip strategy"
                        }})).start()
            raise

    
    '''
    剪辑策略
    '''
    def process_clip_strategy(self, segments, preset, project_no):
        """
        Process clip segments according to preset strategy.
        
        Args:
            segments (list): List of segment dictionaries with start, end, and description
            preset (dict): Preset settings including prompt and language
            
        Returns:
            dict: Standardized transcription format or None if processing fails
        """
        try:
            # Set up default prompts for different languages
            default_prompts = {
                'en': 'Extract the most interesting and important parts of the video',
                'zh': '提取视频中最有趣和最重要的部分',
                'ja': '動画の中で最も面白く重要な部分を抽出する',
            }
            
            # Get language and prompt
            narration_lang = preset.get('narrationLang', 'ja')
            default_prompt = default_prompts.get(narration_lang, default_prompts['ja'])
            clip_desc = preset.get('prompt', default_prompt)
            print(f'剪辑描述: {clip_desc}')

            # Select relevant segments
            if not segments:
                self.logger.warning("No segments found for the video")
                return None

            strategy_info = {
                'input_segments_count': len(segments),
                'strategy_prompt': clip_desc,
                'language': narration_lang,
                'processing_time': 0,
                'selected_clips': None,
                'error': None
            }
            
            start_time = time.time()  # Start timing
            
            try:
                clip_result = self.video_clip_util.select_clips_by_strategy(
                    segments,
                    clip_desc,
                    preset
                )
                strategy_info['selected_clips'] = clip_result
            except Exception as e:
                strategy_info['error'] = str(e)
                raise
            finally:
                elapsed_time = time.time() - start_time
                strategy_info['processing_time'] = elapsed_time
            
            print(f"剪辑策略 took {elapsed_time:.2f} seconds")
            
            if elapsed_time > 300:  # Alert if taking longer than 5 minutes
                self.logger.warning(f"=========> Clip selection took longer than expected: {elapsed_time:.2f} seconds")
            
            if not clip_result:
                self.logger.warning("No clips selected after merging segments")
                return None

            self.logger.info(f'Selected {len(clip_result)} clips from {len(segments)} segments')
            return clip_result

        except Exception as e:
            self.logger.error(f"Error in clip strategy processing: {str(e)}")
            self.logger.error(f"Input segments: {segments}")  # Log the input for debugging
            self.http_util._send_status_callback(self.callback_url, status_data = {
                "data":{
                    "projectNo": project_no,
                },
                "error": {
                    "code": 500,
                    "message":f"Error in clip strategy processing: {str(e)}"
                }
            })
            return None
