import torch
from PIL import Image
import os
import logging
import threading
from typing import List, Dict
from transformers import Blip2Processor, Blip2ForConditionalGeneration

logger = logging.getLogger(__name__)

class BlipAgent:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(BlipAgent, cls).__new__(cls)
            return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.processor = None
            self.model = None
            self.init_models()
            self.initialized = True

    def init_models(self):
        """初始化模型"""
        try:
            logger.info(f"Initializing BLIP2 model on {self.device}")
            self._init_single_model()
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

    def _init_single_model(self):
        """初始化 BLIP2 模型和处理器"""
        try:
            logger.info("Loading BLIP2 processor...")
            self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            
            logger.info(f"Loading BLIP2 model on {self.device}...")
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                device_map="auto",
                torch_dtype=torch.float16  # 使用半精度以节省显存
            )
            
            logger.info("BLIP2 model and processor loaded successfully")
            
        except Exception as e:
            logger.error(f"Error initializing BLIP2 model: {str(e)}")
            raise

    def generate_caption_by_local_path(self, local_frame_path):
        """生成图片描述"""
        try:
            if not os.path.exists(local_frame_path):
                raise FileNotFoundError(f"Image file not found: {local_frame_path}")

            logger.info(f"\n=== Starting caption generation for {os.path.basename(local_frame_path)} ===")
            
            # 加载图片
            image = Image.open(local_frame_path).convert('RGB')
            
            # 准备输入
            question = "What is shown in this image?"
            inputs = self.processor(image, question, return_tensors="pt")
            
            # 将输入移到正确的设备上
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成描述
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=100)
                caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            logger.info(f"Caption generated: {caption}")
            return caption

        except Exception as e:
            logger.error(f"Error generating caption: {str(e)}")
            return None

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
