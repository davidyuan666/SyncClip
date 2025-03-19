import logging
from src.utils.remote_llm_util import RemoteLLMUtil
import subprocess
import os
import time
from typing import List

class WhisperAgent:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.remote_llm_util = RemoteLLMUtil()
  
    def native_chat(self, input_text: str) -> str:
        """
        使用LLM生成回复
        """
        return self.remote_llm_util.native_chat(input_text)
    
    def native_structured_chat(self, input_text: str) -> str:
        """
        使用LLM生成回复
        """
        return self.remote_llm_util.native_structured_chat(input_text)
    
             
    def transcribe_audio_by_whisper(self, audio_path: str,lang:str) -> List[dict]:
        """
        使用语音识别获取音频的时间戳信息
        """
        try:
            # 使用 whisper 或其他语音识别工具获取时间戳
            segments = self.remote_llm_util.transcribe_with_timestamps(audio_path,lang)
            
            return [
                {
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text']
                }
                for segment in segments
            ]
        except Exception as e:
            return []
            
    