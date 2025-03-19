import logging
from threading import Thread
from src.utils.video_clip_util import VideoClipUtil
from src.utils.http_util import HttpUtil
class SegmentAgent:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.callback_url = None
        self.http_util = HttpUtil()
        self.video_clip_util = VideoClipUtil()

    def merge_similar_segments(self, segments, narration_lang, project_no):
        """合并相似片段"""
        try:
            merge_result = self.video_clip_util._merge_similar_segments(
                segments,
                lang=narration_lang
            )
            
            # 如果 merge_result 是字典
            if isinstance(merge_result, dict):
                merged_segments = merge_result.get('transcription', [])
            # 如果 merge_result 直接是列表
            elif isinstance(merge_result, list):
                merged_segments = merge_result
            else:
                raise TypeError(f"Unexpected merge_result type: {type(merge_result)}")
            
            if not merged_segments:
                raise Exception("No segments after merging")
            

            return merged_segments
            
        except Exception as e:
            self.logger.error(f"Failed to merge segments: {str(e)}")
            if self.callback_url:
                Thread(target=self.http_util._send_callback, args=(self.callback_url, {
                        "data":{
                            "projectNo": project_no,
                        },
                        "error": {
                            "code": 500,
                            "message": "Failed to merge segments"
                        }})).start()
            raise