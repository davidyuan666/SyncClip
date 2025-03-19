import logging
import os
import tempfile
import whisper
import openai
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import uuid
from datetime import timedelta,datetime
from src.utils.tencent_cos_util import COSOperationsUtil
from src.utils.llm_util import LLMUtil
import httpx
import ffmpeg
from urllib.parse import urlparse

class VideoAnalyzeAgent:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.whisper_model = whisper.load_model("base")
        self.llm_util = LLMUtil()
        self.cos_ops_util = COSOperationsUtil()  # 假设你已经有这个工具类
        self.http_client = None  # 初始化为None


    async def _ensure_http_client(self):
        """确保 HTTP 客户端已初始化"""
        if not self.http_client:
            self.http_client = httpx.AsyncClient()
        return self.http_client


    async def analyze_video(self, data: dict) -> dict:
        """
        分析视频内容并生成报告
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            dict: 包含分析结果和PDF URL的字典
        """
        try:
            # 1. 提取音频
            cos_video_url = data.get('video_url')
            callback_url = data.get('callback_url')

            # 1. 下载视频
            local_video_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.mp4")
            self.logger.info(f"Downloading cos video from {cos_video_url}")
            
            try:
                if 'https' in cos_video_url:
                    parsed_url = urlparse(cos_video_url)
                    remote_path = parsed_url.path.lstrip('/')
                    self.cos_ops_util.download_file(remote_path, local_video_path)
            except Exception as e:
                raise Exception(f"Failed to download video: {str(e)}")

            self.logger.info(f"Extracting audio from {local_video_path}")
            audio_path = await self._extract_audio(local_video_path)
            
            # 2. 使用Whisper进行语音识别
            self.logger.info(f"Transcribing audio from {audio_path}")
            transcription_result = await self._transcribe_audio(audio_path)
            
            # 3. 使用LLM分析内容
            self.logger.info(f"Analyzing content from {transcription_result}")
            analysis_result = await self._analyze_content(transcription_result)
            
            # 4. 生成PDF报告
            self.logger.info(f"Generating PDF report from {transcription_result} and {analysis_result}")
            pdf_path = await self._generate_pdf_report(transcription_result, analysis_result)
            
            # 5. 上传PDF到COS
            self.logger.info(f"Uploading PDF to COS from {pdf_path}")
            pdf_url = await self._upload_to_cos(pdf_path)
            
            # 6. 清理临时文件
            self.logger.info(f"Cleaning up temporary files from {audio_path} and {pdf_path}")
            self._cleanup_files(audio_path, pdf_path)
            
            result = {
                "status": "success",
                "pdf_url": pdf_url,
                "summary": analysis_result["summary"]
            }
            self.logger.info(f"Sending callback to {callback_url} with result: {result}")
        
            await self._send_callback(callback_url,result)

            return result
            
        except Exception as e:
            error_result = {
                "status": "error",
                "video_url": cos_video_url,
                "error_message": str(e)
            }
            # 即使发生错误也尝试发送回调
            try:
                if callback_url:
                    await self._send_callback(callback_url,error_result)
            except Exception as callback_error:
                self.logger.error(f"Failed to send error callback: {str(callback_error)}")
            
            self.logger.error(f"Error analyzing video: {str(e)}")
            raise
        
        finally:
            self._cleanup_files(local_video_path,audio_path)
    

    async def _send_callback(self, callback_url: str, result: dict) -> None:
        """发送分析结果回调"""
        try:
            client = await self._ensure_http_client()
            response = await client.post(
                callback_url,
                json={
                    "timestamp": datetime.now().isoformat(),
                    "data": result
                },
                timeout=30.0
            )
            response.raise_for_status()
            self.logger.info(f"Successfully sent callback for project: {result.get('project_no')}")
            
        except httpx.RequestError as e:
            error_msg = f"Network error when sending callback: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
            
        except httpx.HTTPStatusError as e:
            error_msg = f"Callback server returned error status {e.response.status_code}: {e.response.text}"
            self.logger.error(error_msg)
            raise Exception(error_msg)


    async def _extract_audio(self, video_path: str) -> str:
        """
        使用ffmpeg提取视频中的音频，使用GPU加速
        """
        try:
            temp_audio_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.mp3")
            
            # 配置ffmpeg命令
            stream = ffmpeg.input(video_path, hwaccel='cuda')
            
            
            # 输出配置只包含音频编码参数
            stream = ffmpeg.output(
                stream.audio,
                temp_audio_path,
                acodec='libmp3lame',
                ac=2,  # 双声道
                ar='44100'  # 采样率
            )
            
            # 运行ffmpeg命令
            self.logger.info(f"Extracting audio using ffmpeg with GPU acceleration")
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            
            if not os.path.exists(temp_audio_path):
                raise Exception("Audio extraction failed: output file not created")
                
            return temp_audio_path
            
        except ffmpeg.Error as e:
            error_message = f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}"
            self.logger.error(error_message)
            raise Exception(error_message)
        except Exception as e:
            raise Exception(f"Failed to extract audio: {str(e)}")



    async def _transcribe_audio(self, audio_path: str) -> list:
        """使用Whisper进行语音识别"""
        try:
            result = self.whisper_model.transcribe(audio_path)
            # 格式化时间戳和文本
            segments = []
            for segment in result["segments"]:
                segments.append({
                    "start": str(timedelta(seconds=int(segment["start"]))),
                    "end": str(timedelta(seconds=int(segment["end"]))),
                    "text": segment["text"].strip()
                })
            return segments
        except Exception as e:
            raise Exception(f"Failed to transcribe audio: {str(e)}")

    async def _analyze_content(self, transcription: list) -> dict:
        """使用LLM分析转录内容"""
        try:
            # 将转录文本合并
            full_text = " ".join([segment["text"] for segment in transcription])
            
            analyze_prompt = f"""
            请分析以下转录内容，并根据转录的内容的主要内容，按照以下格式生成一个简洁的摘要。
            
            转录内容:
            {full_text}
            
            摘要格式:
            1. 主要内容
            2. 核心点
            3. 最终结论
            
            """
            # 使用 LLMUtil 的同步方法，不使用 await
            response = self.llm_util.native_chat(analyze_prompt)
            
            return {
                "summary": response,
                "analyzed_at": datetime.now().isoformat()
            }
        except Exception as e:
            error_msg = f"Failed to analyze content: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)

    async def _generate_pdf_report(self, transcription: list, analysis: dict) -> str:
        """生成PDF报告"""
        try:
            pdf_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.pdf")
            doc = SimpleDocTemplate(pdf_path, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            # 添加标题
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30
            )
            story.append(Paragraph("视频内容分析报告", title_style))
            story.append(Spacer(1, 20))

            # 添加分析摘要
            story.append(Paragraph("内容摘要", styles['Heading2']))
            story.append(Paragraph(analysis["summary"], styles['Normal']))
            story.append(Spacer(1, 20))

            # 添加详细转录
            story.append(Paragraph("详细转录", styles['Heading2']))
            for segment in transcription:
                time_text = f"[{segment['start']} - {segment['end']}]"
                story.append(Paragraph(f"{time_text}: {segment['text']}", styles['Normal']))
                story.append(Spacer(1, 10))

            doc.build(story)
            return pdf_path
        except Exception as e:
            raise Exception(f"Failed to generate PDF report: {str(e)}")

    async def _upload_to_cos(self, pdf_path: str) -> str:
        """上传PDF到COS"""
        try:
            filename = os.path.basename(pdf_path)
            remote_path = f"reports/{str(uuid.uuid4())}_{filename}"
            
            self.cos_ops_util.upload_file(
                local_file_path=pdf_path,
                cos_file_path=remote_path
            )
            
            return self.cos_ops_util.get_file_url(remote_path)
        except Exception as e:
            raise Exception(f"Failed to upload PDF to COS: {str(e)}")

    def _cleanup_files(self, *file_paths):
        """清理临时文件"""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                self.logger.warning(f"Failed to delete temporary file {file_path}: {str(e)}")