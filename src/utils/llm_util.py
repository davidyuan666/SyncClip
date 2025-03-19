import base64
import json  # Used for working with JSON data
import os
import tempfile
import uuid
import torch
import whisper
import requests
from openai import AzureOpenAI, OpenAI
from typing import List, Optional 
import logging


class LLMUtil:
    def __init__(self):
        self.vision_model = "gpt-4-vision-preview"
        self.api_base_url = os.getenv("API_BASE_URL").rstrip('/')
        self.model = os.getenv("MODEL")
        self.struct_model = os.getenv("STRUCT_MODEL")
        self.whisper_model = None
        self.whisper_model_size = os.getenv("WHISPER_MODEL_SIZE")
        self.active_key = [
            {
                'key': os.getenv("OPENAI_ACTIVE_KEY"),
                'name': 'seemingai_key',
                'status': True
            }
        ]
        self.openai_client = OpenAI(api_key=self.active_key[0]['key'])


        """
        OpenAI 配置
        """
        self.api_keys = [
            {
                "key": os.getenv("OPENAI_API_KEY"),
                "name": "seemingai_key",
                "status": True,
            }
        ]

        self.current_key_index = 0

        # 初始化使用第一个活跃的key
        active_key = next((key for key in self.api_keys if key["status"]), None)
        if not active_key:
            raise Exception("No active API keys available")

        self.openai_client = OpenAI(api_key=active_key["key"])

        self.openai_chat_url = "https://api.openai.com/v1/chat/completions"
        self.messages = []

        """
        Azure 配置
        """
        self.azure_key = "c5cf096ac45e477ca8766c389299d243"
        self.azure_endpoint = "https://vesync-openai-survey.openai.azure.com/"

        self.azure_client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.azure_key,
            api_version="2024-02-01",
        )

        self.azure_json_client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.azure_key,
            api_version="2024-03-01-preview",
        )

        """
        eleven lab 配置 https://elevenlabs.io/docs/api-reference/getting-started
        """
        self.XI_API_KEY = os.getenv("XI_API_KEY")

        """
        milvus 配置
        """
        self.milvus_key = os.getenv("MILVUS_KEY")

        self.logger = logging.getLogger(__name__)
        

    def get_voice_id(self):
        """
        获取 ElevenLabs 可用的语音列表

        Returns:
            dict: 语音名称和ID的映射字典，失败时返回空字典
        """
        try:
            url = "https://api.elevenlabs.io/v1/voices"

            headers = {
                "Accept": "application/json",
                "xi-api-key": self.XI_API_KEY,
                "Content-Type": "application/json",
            }

            self.logger.info("Fetching available voices from ElevenLabs...")
            response = requests.get(url, headers=headers)

            if not response.ok:
                self.logger.error(f"Failed to fetch voices: {response.text}")
                return {}

            data = response.json()

            # 创建语音映射字典
            voice_map = {}
            for voice in data.get("voices", []):
                name = voice.get("name")
                voice_id = voice.get("voice_id")
                if name and voice_id:
                    voice_map[name] = {
                        "id": voice_id,
                        "description": voice.get("description", ""),
                        "preview_url": voice.get("preview_url", ""),
                        "labels": voice.get("labels", {}),
                        "category": voice.get("category", "unknown"),
                    }

            # 打印可用的语音列表
            for name, info in voice_map.items():
                self.logger.info(f"- {name}: {info['id']} ({info['category']})")

            return voice_map

        except Exception as e:
            self.logger.error(f"Error fetching voices: {str(e)}")
            return {}

    def get_default_voice_id(self):
        """
        获取默认语音ID

        Returns:
            str: 默认语音ID，如果获取失败则返回预设的备用ID
        """
        try:
            voices = self.get_voice_id()

            # 优先选择中文语音
            preferred_voices = [
                "Chinese Male 1",
                "Chinese Female 1",
                "Multilingual Male 1",
                "Multilingual Female 1",
            ]

            for voice_name in preferred_voices:
                if voice_name in voices:
                    voice_id = voices[voice_name]["id"]
                    self.logger.info(f"Using voice: {voice_name} ({voice_id})")
                    return voice_id

            # 如果没有找到首选语音，使用第一个可用的语音
            if voices:
                first_voice = next(iter(voices.values()))
                self.logger.info(f"Using first available voice: {first_voice['id']}")
                return first_voice["id"]

            # 如果获取失败，使用备用ID
            fallback_id = "21m00Tcm4TlvDq8ikWAM"
            self.logger.info(f"Using fallback voice ID: {fallback_id}")
            return fallback_id

        except Exception as e:
            self.logger.error(f"Error getting default voice ID: {str(e)}")
            return "21m00Tcm4TlvDq8ikWAM"  # 备用ID

    def text_to_speech_by_elevenlabs(self, content, voice_id):
        """
        使用 ElevenLabs API 将文本转换为语音

        Args:
            content (str): 要转换的文本内容

        Returns:
            str: 输出音频文件的路径，失败时返回 None
        """
        try:
            CHUNK_SIZE = 1024

            # 创建临时文件，确保每次生成唯一的文件名
            temp_dir = os.path.join(os.getcwd(), "temp", "audio")
            os.makedirs(temp_dir, exist_ok=True)
            output_path = os.path.join(temp_dir, f"speech_{uuid.uuid4()}.mp3")

            # API endpoint
            tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"

            headers = {"Accept": "application/json", "xi-api-key": self.XI_API_KEY}

            data = {
                "text": content,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.8,
                    "style": 0.0,
                    "use_speaker_boost": True,
                },
            }

            self.logger.info(f"Sending TTS request for text: {content[:100]}...")
            response = requests.post(tts_url, headers=headers, json=data, stream=True)

            if response.ok:
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)

                self.logger.info(f"Audio saved successfully to: {output_path}")
                return output_path

            else:
                error_msg = f"Failed to generate audio: {response.text}"
                self.logger.error(error_msg)
                return None

        except Exception as e:
            self.logger.error(f"Error in text_to_speech_by_elevenlabs: {str(e)}")
            return None

    """
    openai native chat 接口
    """

    def native_chat(self, message):
        """
        Generate video clip instructions

        Args:
            message (str): User's clip requirements description
            object_define (dict): OpenAI response format definition

        Returns:
            dict: Dictionary containing clip steps and final answer
            None: If processing fails
        """
        try:
            self.messages = []
            self.messages.append({"role": "user", "content": message})
            completion = self.openai_client.chat.completions.create(
                model=self.model, messages=self.messages
            )

            response = completion.choices[0].message.content
            return response

        except Exception as e:
            return f"Error during native chat completion: {e}"

    """
    openai structured chat 接口
    """

    def native_structured_chat(self, message, object_define):
        """
        Generate video clip instructions

        Args:
            message (str): User's clip requirements description
            object_define (dict): OpenAI response format definition

        Returns:
            dict: Dictionary containing clip steps and final answer
            None: If processing fails
        """
        try:
            # Clear previous messages if any
            self.messages = []

            # Add system and user messages
            self.messages.extend(
                [
                    {
                        "role": "system",
                        "content": "You are a professional video editing expert who can provide precise editing suggestions and operational steps based on user requirements.",
                    },
                    {"role": "user", "content": message},
                ]
            )

            

            completion = self.openai_client.beta.chat.completions.parse(
                model=self.struct_model,
                messages=self.messages,
                response_format=object_define,
            )
            self.logger.info(f"completion: {completion}")
            response = completion.choices[0].message
            self.logger.info("Successfully generated clip instructions")
            return response

        except Exception as e:
            self.logger.error(f"Error generating clip instructions: {str(e)}")
            return None

    """
    azure chat 接口
    """

    def chat(self, message):
        self.messages = []
        self.messages.append({"role": "user", "content": message})
        try:
            chat_response = self.azure_client.chat.completions.create(
                model=self.model, messages=self.messages, temperature=0
            )
            response_message = chat_response.choices[0].message.content
            return response_message
        except Exception as e:
            return f"Error during chat completion: {e}"

    """
    azure structured chat 接口
    """

    def structured_chat(self, message):
        """
        Sends a message to the chatbot and receives a response.

        :param message: User input message
        :return: Chatbot's response message
        """
        self.messages = []
        self.messages.append(
            {
                "role": "system",
                "content": "You are a helpful assistant designed to output JSON.",
            },
        )
        self.messages.append({"role": "user", "content": message})
        try:
            json_response = self.azure_json_client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                response_format={"type": "json_object"},
            )
            response_message = json_response.choices[0].message.content
            self.messages.append({"role": "assistant", "content": response_message})
            return response_message
        except Exception as e:
            return f"Error during chat completion: {e}"

    def audio_to_text(self, audio_path):
        with open(audio_path, "rb") as audio_file:
            transcription = self.openai_client.audio.transcriptions.create(
                model="whisper-1", file=audio_file
            )
        self.logger.info(f"Transcription: {transcription.text}")
        return transcription.text

    def audio_to_srt(self, audio_path):
        with open(audio_path, "rb") as audio_file:
            transcription = self.openai_client.audio.transcriptions.create(
                model="whisper-1", file=audio_file, response_format="srt"
            )
        return transcription

    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _calculate_token_price(self, input_tokens, output_tokens):
        if self.model == "gpt-4o-mini":
            input_cost = input_tokens * (0.150 / 1000000)
            output_cost = output_tokens * (0.600 / 1000000)
            return float(input_cost + output_cost)
        # Add more models and their pricing here if needed
        return 0  # Default return if model not recognized

    def analyze_image(self, image_path, prompt=None, max_tokens=300):
        """
        使用 OpenAI Vision 模型分析图片

        Args:
            image_path (str): 图片文件路径
            prompt (str, optional): 自定义提示词。默认为简单描述图片内容
            max_tokens (int, optional): 返回结果的最大token数。默认300

        Returns:
            str: 分析结果
            None: 如果处理失败
        """
        try:
            # 如果没有提供提示词，使用默认提示词
            if not prompt:
                prompt = "请描述这张图片中的内容，包括重要的视觉元素和细节。"

            # 将图片转换为 base64
            base64_image = self._encode_image(image_path)

            # 构建请求消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ]

            # 调用 API
            response = self.openai_client.chat.completions.create(
                model=self.vision_model, messages=messages, max_tokens=max_tokens
            )

            return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"分析图片时发生错误: {str(e)}")
            return None

    def analyze_multiple_images(self, image_paths, prompt=None, max_tokens=500):
        """
        同时分析多张图片

        Args:
            image_paths (list): 图片文件路径列表
            prompt (str, optional): 自定义提示词
            max_tokens (int, optional): 返回结果的最大token数

        Returns:
            str: 分析结果
            None: 如果处理失败
        """
        try:
            if not prompt:
                prompt = "请分析这些图片，描述它们的内容并指出它们之间的关系或差异。"

            # 构建包含多张图片的消息内容
            content = [{"type": "text", "text": prompt}]

            # 添加所有图片
            for image_path in image_paths:
                base64_image = self._encode_image(image_path)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    }
                )

            # 发送请求
            response = self.openai_client.chat.completions.create(
                model=self.vision_model,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"分析多张图片时发生错误: {str(e)}")
            return None


    def _load_whisper_model(self):
        """
        懒加载 Whisper 模型，使用 GPU 并优化性能，包含内存管理和错误恢复
        """
        try:
            if self.whisper_model is None:
                self.logger.info(f"Loading Whisper model ({self.whisper_model_size})...")
                
                # 检查 CUDA 是否可用
                device = "cuda" if torch.cuda.is_available() else "cpu"
                if device == "cuda":
                    # 获取 GPU 信息
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    free_memory = torch.cuda.mem_get_info()[0] / 1024**3
                    self.logger.info(f"Using GPU: {gpu_name} with {gpu_memory:.2f}GB memory, {free_memory:.2f}GB free")
                    
                    # 设置 PyTorch 内存分配器配置
                    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
                    
                    # 如果可用显存太少，直接使用 CPU
                    if free_memory < 1.0:  # 如果可用显存小于1GB
                        self.logger.warning("Insufficient GPU memory, falling back to CPU")
                        device = "cpu"
                    else:
                        # 启用 cuDNN 自动调优
                        torch.backends.cudnn.benchmark = True
                        # 使用确定性算法
                        torch.backends.cudnn.deterministic = True

                # 尝试加载模型
                try:
                    self.whisper_model = whisper.load_model(
                        self.whisper_model_size,
                        device=device
                    )
                    
                    # 如果是 GPU 且内存较少，使用半精度
                    if device == "cuda" and free_memory < 2.0:
                        self.logger.info("Converting model to half precision to save memory")
                        self.whisper_model = self.whisper_model.half()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e) and device == "cuda":
                        self.logger.warning("GPU OOM, falling back to CPU")
                        # 清理 GPU 内存并重试使用 CPU
                        if hasattr(self, 'whisper_model'):
                            del self.whisper_model
                        torch.cuda.empty_cache()
                        device = "cpu"
                        self.whisper_model = whisper.load_model(
                            self.whisper_model_size,
                            device=device
                        )
                    else:
                        raise e

                self.logger.info(f"Whisper model loaded successfully on {device}")
                
                # 设置模型配置
                self.whisper_config = {
                    "temperature": 0,
                    "no_speech_threshold": 0.6,
                    "logprob_threshold": -1.0,
                    "compression_ratio_threshold": 2.4,
                    "condition_on_previous_text": True,
                }
                
                return True
                    
        except Exception as e:
            self.logger.error(f"Error loading Whisper model: {str(e)}")
            if torch.cuda.is_available():
                free_memory = torch.cuda.mem_get_info()[0] / 1024**3
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.logger.error(f"GPU Memory Status - Total: {total_memory:.2f}GB, Free: {free_memory:.2f}GB")
            
            # 确保在错误情况下模型为 None
            self.whisper_model = None
            return False
        


    def transcribe_with_timestamps(self, audio_path: str) -> List[dict]:
        """
        使用 Whisper 模型进行语音识别并获取时间戳信息
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            List[dict]: 包含时间戳和文本的段落列表
        """
        try:
            self._load_whisper_model()
            
            # 转写音频文件
            result = self.whisper_model.transcribe(
                audio_path,
                language="ja",  # 可以根据需要设置语言
                task="transcribe",
                verbose=True
            )
            
            # 提取段落信息
            segments = []
            for segment in result["segments"]:
                segments.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'].strip()
                })
                
            self.logger.info(f"Successfully transcribed {len(segments)} segments")
            return segments
            
        except Exception as e:
            self.logger.error(f"Error in audio transcription: {str(e)}")
            return []

    