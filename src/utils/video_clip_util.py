import tempfile
from src.utils.remote_llm_util import RemoteLLMUtil
from src.utils.llm_util import LLMUtil
from src.utils.tencent_cos_util import COSOperationsUtil
import re
import uuid
import os
import json
import tempfile
import shutil
import logging
from PIL import Image, ImageDraw, ImageFont
import cv2
from pydantic import BaseModel, Field
import ffmpeg
import numpy as np
import cv2
from typing import Optional, Dict, List, Tuple,Any,Union
import subprocess
from pathlib import Path
import torch

class Transcription(BaseModel):
    """音频文本片段模型"""
    start: float = Field(..., description="开始时间（秒）")
    end: float = Field(..., description="结束时间（秒）")
    text: str = Field(..., description="文本内容")

class TranscriptionResponse(BaseModel):
    """音频文本响应模型"""
    transcription: List[Transcription] = Field(..., description="选中的音频文本片段列表")

    class Config:
        json_schema_extra = {
            "example": {
                "transcription": [
                    {
                        "start": 0.0,
                        "end": 11.0,
                        "text": "示例文本内容"
                    }
                ]
            }
        }




'''
sudo apt-get install fonts-wqy-zenhei fonts-wqy-microhei
'''
class VideoClipUtil:
    def __init__(self):
        self.remote_llm_util = RemoteLLMUtil()
        self.llm_util = LLMUtil()
        self.cos_util = COSOperationsUtil()        
        self.base_cos_url = os.getenv('BASE_COS_URL')
        self.logger = logging.getLogger(__name__)


    '''
    具体的剪辑策略
    '''
    def select_clips_by_strategy(self, segments, prompt, preset):
        """
        选择并合并视频和音频分析的片段，分三步进行：
        1. 重写提示词生成简单剧本
        2. 合并相似片段
        3. 基于剧本选择最终片段
        
        Args:
            segments (list): 包含视频和音频分析结果的片段列表
            prompt (str): 用户提示词
            preset (dict): 预设配置
            
        Returns:
            list: 合并后的片段列表，每个片段包含 start, end, text 信息
        """
        try:
            lang = preset.get('narrationLang', 'ja')
            '''
            剪辑模板
            '''
            # template = preset.get('template',1)
            
            selected_clips = self._select_clips_by_script(segments, prompt, lang)
            
            return selected_clips

        except Exception as e:
            self.logger.info(f"Error in select_clips_by_strategy: {str(e)}")
            return None
        

    def _generate_script(self, prompt, lang):
        """
        生成视频剧本
        
        Args:
            prompt (str): 原始描述文本
            lang (str): 语言代码 ('en', 'ja', 'zh')
            
        Returns:
            str: 生成的剧本文本
        """
        rewrite_prompts = {
            'en': f'''As a professional video script writer, create a compelling script based on this description:
    "{prompt}"

    Please follow these guidelines:
    1. Core Elements:
    - Identify and emphasize main themes
    - Focus on key messages and emotions
    - Maintain clear narrative structure

    2. Style Requirements:
    - Use concise, vivid language
    - Create smooth scene transitions
    - Ensure natural flow of ideas

    3. Technical Considerations:
    - Keep descriptions visual and actionable
    - Consider pacing and timing
    - Maintain consistent tone

    Please provide a script that is:
    - Engaging yet concise
    - Well-structured
    - Suitable for video production

    Output the script in a clear, professional format.''',

            'ja': f'''プロのビデオ脚本家として、以下の説明に基づいて魅力的な脚本を作成してください：
    "{prompt}"

    以下のガイドラインに従ってください：
    1. 核となる要素：
    - メインテーマの特定と強調
    - 重要なメッセージと感情に焦点を当てる
    - 明確な物語構造の維持

    2. スタイル要件：
    - 簡潔で生き生きとした言葉を使用
    - スムーズなシーン転換
    - 自然なアイデアの流れ

    3. 技術的考慮事項：
    - 視覚的で実行可能な描写
    - ペースとタイミングの考慮
    - 一貫したトーンの維持

    以下の要件を満たす脚本を提供してください：
    - 魅力的かつ簡潔
    - 構造が整っている
    - 映像制作に適している

    プロフェッショナルな形式で脚本を出力してください。''',

            'zh': f'''作为专业视频脚本作家，请基于以下描述创作一个引人入胜的剧本：
    "{prompt}"

    请遵循以下指南：
    1. 核心要素：
    - 识别并强调主要主题
    - 突出关键信息和情感
    - 保持清晰的叙事结构

    2. 风格要求：
    - 使用简洁生动的语言
    - 创造流畅的场景转换
    - 确保思路自然流畅

    3. 技术考虑：
    - 保持描述的可视性和可执行性
    - 注意节奏和时间把控
    - 维持一致的语气

    请提供满足以下要求的剧本：
    - 引人入胜且简洁
    - 结构完整
    - 适合视频制作

    请以清晰、专业的格式输出剧本。'''
        }

        try:
            # 获取对应语言的提示词
            rewrite_prompt = rewrite_prompts.get(lang)
            if not rewrite_prompt:
                self.logger.warning(f"Language {lang} not supported, falling back to Chinese")
                rewrite_prompt = rewrite_prompts['zh']

            # 调用 LLM 生成剧本
            response = self.remote_llm_util.native_chat(rewrite_prompt)

            # 处理响应
            if not response or not response.strip():
                raise ValueError("Generated script is empty")

            # 清理和格式化剧本
            script = self._format_script(response.strip(), lang)
            
            self.logger.info(f"Successfully generated script in {lang}")
            return script

        except Exception as e:
            self.logger.error(f"Error generating script: {str(e)}")
            # 返回一个基本的错误提示
            error_messages = {
                'en': "Error generating script. Please try again.",
                'ja': "脚本の生成中にエラーが発生しました。もう一度お試しください。",
                'zh': "生成剧本时出错，请重试。"
            }
            return error_messages.get(lang, error_messages['zh'])

    def _format_script(self, script: str, lang: str) -> str:
        """
        格式化生成的剧本
        
        Args:
            script (str): 原始剧本文本
            lang (str): 语言代码
            
        Returns:
            str: 格式化后的剧本
        """
        try:
            # 移除多余的空行
            script = '\n'.join(line for line in script.splitlines() if line.strip())
            
            # 确保段落之间有适当的间距
            script = re.sub(r'\n{3,}', '\n\n', script)
            
            # 根据语言添加适当的标点符号
            if lang == 'zh':
                # 统一中文标点
                script = re.sub(r'\.{3,}', '……', script)  # 省略号
                script = re.sub(r'!', '！', script)       # 感叹号
                script = re.sub(r'\?', '？', script)      # 问号
            elif lang == 'ja':
                # 统一日文标点
                script = re.sub(r'\.{3,}', '…', script)   # 省略号
                script = re.sub(r'!', '！', script)       # 感叹号
                script = re.sub(r'\?', '？', script)      # 问号
                
            return script.strip()
            
        except Exception as e:
            self.logger.error(f"Error formatting script: {str(e)}")
            return script  # 返回原始文本


    def _merge_similar_segments(self, segments, lang='zh'):
        """
        基于场景智能合并视频片段，将相似内容的片段组合成完整场景
        
        Args:
            segments (list): 待合并的片段列表
            lang (str): 语言代码 'zh'/'en'/'ja'
        
        Returns:
            list: 合并后的场景片段列表
        """
        if not segments:
            return []
                
        # 1. 预处理：按时间排序
        sorted_segments = sorted(segments, key=lambda x: x['start'])
        
        # 2. 准备片段文本
        segments_text = "\n\n".join([
            f"片段 {i+1}:\n"
            f"开始时间: {seg['start']:.2f}秒\n"
            f"结束时间: {seg['end']:.2f}秒\n"
            f"内容: {seg['text']}\n"
            for i, seg in enumerate(sorted_segments)
        ])

        self.logger.info(f"待处理的视频片段: {segments_text}")

        prompts = {
            'zh': f"""作为视频场景分析专家，请将以下时间片段合并为完整的场景描述：

    输入片段:
    {segments_text}

    分析要求：
    1. 场景识别和合并：
    - 识别内容相关的片段，将它们合并为完整场景
    - 确保场景之间有明显的主题或内容转换
    - 合并时保持时间的连续性

    2. 场景内容处理：
    - 提供完整、连贯的场景描述
    - 补充必要的场景细节和上下文
    - 删除重复或冗余的信息
    - 使用更生动的语言描述场景

    3. 时间轴处理：
    - 使用最早的开始时间和最晚的结束时间
    - 确保场景时间完整覆盖原始片段
    - 避免场景之间的时间间隔

    请返回JSON格式的场景列表：
    {{
        "transcription": [
            {{
                "start": 开始时间(数字),
                "end": 结束时间(数字),
                "text": "完整的场景描述"
            }},
            ...
        ]
    }}

    注意事项：
    1. 场景描述要完整且富有细节
    2. 时间必须准确对应原始片段
    3. 场景之间要有清晰的过渡
    4. 保持叙述的流畅性和连贯性""",

            'en': f"""As a video scene analysis expert, please merge these time segments into complete scene descriptions:

    Input segments:
    {segments_text}

    Analysis Requirements:
    1. Scene Recognition and Merging:
    - Identify related content and merge into complete scenes
    - Ensure clear thematic or content transitions between scenes
    - Maintain temporal continuity during merging

    2. Scene Content Processing:
    - Provide complete, coherent scene descriptions
    - Add necessary scene details and context
    - Remove duplicate or redundant information
    - Use vivid language for scene descriptions

    3. Timeline Processing:
    - Use earliest start and latest end times
    - Ensure complete coverage of original segments
    - Avoid gaps between scenes

    Please return scenes in JSON format:
    {{
        "transcription": [
            {{
                "start": start_time(number),
                "end": end_time(number),
                "text": "complete scene description"
            }},
            ...
        ]
    }}

    Important Notes:
    1. Scene descriptions should be complete and detailed
    2. Times must accurately match original segments
    3. Clear transitions between scenes
    4. Maintain narrative flow and coherence""",

            'ja': f"""映像シーン分析の専門家として、以下の時間セグメントを完全なシーン描写にまとめてください：

    入力セグメント:
    {segments_text}

    分析要件：
    1. シーン認識とマージ：
    - 関連コンテンツを識別し、完全なシーンにまとめる
    - シーン間で明確なテーマや内容の転換を確保
    - マージ時に時間の連続性を維持

    2. シーンコンテンツの処理：
    - 完全で一貫性のあるシーン描写を提供
    - 必要なシーンの詳細やコンテキストを追加
    - 重複や冗長な情報を削除
    - 生き生きとした言葉でシーンを描写

    3. タイムライン処理：
    - 最も早い開始時間と最も遅い終了時間を使用
    - 元のセグメントを完全にカバー
    - シーン間のギャップを避ける

    以下のJSON形式でシーンを返してください：
    {{
        "transcription": [
            {{
                "start": 開始時間（数値）,
                "end": 終了時間（数値）,
                "text": "完全なシーン描写"
            }},
            ...
        ]
    }}

    重要な注意点：
    1. シーン描写は完全で詳細であること
    2. 時間は元のセグメントと正確に一致すること
    3. シーン間の転換が明確であること
    4. 物語の流れと一貫性を維持すること

    補足要件：
    - 各シーンの雰囲気や感情も含めて描写
    - 視聴者が映像を明確にイメージできる表現を使用
    - 重要な動きや変化を見落とさない
    - 日本語として自然で分かりやすい表現を使用"""
    
        }

        try:
            # 3. 使用 LLM 处理场景合并
            result = self.remote_llm_util.native_structured_chat(
                prompts.get(lang, prompts['ja'])
            )
            
            self.logger.info(f"场景合并结果: {result}")
            
            # 4. 验证和处理结果
            if not result or 'transcription' not in result:
                raise ValueError("LLM返回的结果格式无效")
                
            validated_scenes = []
            for scene in result['transcription']:
                # 验证时间的有效性
                if not isinstance(scene.get('start'), (int, float)) or \
                not isinstance(scene.get('end'), (int, float)):
                    self.logger.warning(f"场景时间格式无效: {scene}")
                    continue
                    
                duration = scene['end'] - scene['start']
                if duration <= 0:
                    self.logger.warning(f"场景持续时间无效: {duration}秒")
                    continue
                    
                # 验证描述的有效性
                if not scene.get('text') or \
                len(scene['text'].strip()) < 5:  # 确保描述足够详细
                    self.logger.warning(f"场景描述过短或无效: {scene.get('text')}")
                    continue
                    
                validated_scenes.append({
                    'start': float(scene['start']),
                    'end': float(scene['end']),
                    'text': scene['text'].strip()
                })
            
            if not validated_scenes:
                raise ValueError("没有有效的场景")
                
            # 5. 确保场景时间连续性
            validated_scenes = sorted(validated_scenes, key=lambda x: x['start'])
            for i in range(len(validated_scenes)-1):
                if validated_scenes[i]['end'] > validated_scenes[i+1]['start']:
                    # 处理时间重叠
                    mid_point = (validated_scenes[i]['end'] + validated_scenes[i+1]['start']) / 2
                    validated_scenes[i]['end'] = mid_point
                    validated_scenes[i+1]['start'] = mid_point
                    self.logger.info(f"调整重叠场景时间: {validated_scenes[i]['end']} -> {validated_scenes[i+1]['start']}")
            

            self.logger.info(f"合并后的场景: {validated_scenes}")

            return validated_scenes
                
        except Exception as e:
            self.logger.error(f"合并场景时出错: {str(e)}")
            raise


    def _remove_time_overlaps(self, segments):
        """
        移除时间重叠的片段，保留描述更好的版本
        """
        if not segments:
            return []
            
        non_overlapping = [segments[0]]
        
        for current in segments[1:]:
            last = non_overlapping[-1]
            
            # 检查是否有重叠
            if current['start'] < last['end']:
                # 如果有重叠，保留描述更详细的片段
                if len(current['text']) > len(last['text']):
                    non_overlapping[-1] = current
            else:
                non_overlapping.append(current)
        
        return non_overlapping

    def _validate_segments(self, segments):
        """
        验证处理后的片段是否符合要求
        """
        validated = []
        last_end = 0
        
        for seg in segments:
            # 确保时间格式正确
            start = float(seg['start'])
            end = float(seg['end'])
            
            # 验证时间顺序和间隔
            if start >= end:
                continue
            if start < last_end:
                continue
                
            # 确保最小持续时间
            if end - start < 2:
                continue
                
            validated.append({
                'start': round(start, 2),
                'end': round(end, 2),
                'text': seg['text']
            })
            
            last_end = end
        
        return validated


    def _select_clips_by_script(self, segments, prompt, lang):
        """基于剧本选择最终片段"""
        try:
            if not segments:
                self.logger.warning("No segments provided for script selection")
                return None

            # 记录输入数据结构
            self.logger.debug(f"Input segments structure: {segments[:1]}")
                
            segments_text = "\n\n".join([
                f"开始时间点: {seg.get('start', 0)}秒\n"
                f"结束时间点: {seg.get('end', 0)}秒\n"
                f"内容: {seg.get('text', '')}\n"
                for seg in segments
            ])

            # 更新提示词
            analysis_prompts = {
                'zh': f"""作为视频剪辑专家，请根据以下内容创建一个精简的视频剪辑脚本：

            输入信息：
            1. 用户需求：
            {prompt}

            2. 可用片段：
            {segments_text}

            剪辑要求：
            1. 选择多个最能表达用户需求的片段
            2. 确保片段之间的转场自然流畅
            3. 保持画面连贯性和故事性
            4. 避免重复或相似的场景

            请以JSON数组格式返回选中的多个片段，每个元素包含：
            {{
                "start": 数字,  # 开始时间（秒）
                "end": 数字,    # 结束时间（秒）
                "text": "描述"  # 场景描述
            }}

            注意：
            1. 时间必须准确对应原片段
            2. 场景描述要简洁清晰
            3. 确保选择的片段能完整表达主题""",
                
                'ja': f"""映像編集の専門家として、以下の内容に基づいて簡潔な編集脚本を作成してください：

            入力情報：
            1. ユーザーの要望：
            {prompt}

            2. 利用可能なセグメント：
            {segments_text}

            編集要件：
            1. ユーザーの要望を最もよく表現するセグメントを複数選択
            2. シーン転換の自然な流れを確保
            3. 映像の一貫性とストーリー性を維持
            4. 重複や類似シーンを避ける

            以下のJSON配列形式で複数選択したセグメントを返してください：
            {{
                "start": 数値,  # 開始時間（秒）
                "end": 数値,    # 終了時間（秒）
                "text": "説明"  # シーン説明
            }}

            注意：
            1. 時間は元のセグメントと正確に対応すること
            2. シーン説明は簡潔明瞭であること
            3. 選択したセグメントがテーマを完全に表現できること""",
                
                'en': f"""As a video editing expert, please create a concise editing script based on the following:

            Input Information:
            1. User Requirements:
            {prompt}

            2. Available Segments:
            {segments_text}

            Editing Requirements:
            1. Select multiple segments that best express user needs
            2. Ensure smooth transitions between segments
            3. Maintain visual coherence and storytelling
            4. Avoid repetitive or similar scenes

            Please return multiple selected segments in JSON array format, each containing:
            {{
                "start": number,  # Start time in seconds
                "end": number,    # End time in seconds
                "text": "description"  # Scene description
            }}

            Note:
            1. Times must accurately match original segments
            2. Scene descriptions should be clear and concise
            3. Selected segments must fully express the theme"""
            }

            analysis_prompt = analysis_prompts.get(lang, analysis_prompts['ja'])
            result = self.remote_llm_util.native_structured_chat(analysis_prompt)
            self.logger.info(f"Analysis result: {result}")

            if 'transcription' in result:   
                return result.get('transcription', [])
            else:
                return []
    

        except Exception as e:
            self.logger.error(f"Error in _select_clips_by_script: {str(e)}")
            return None
    




    def create_text_frame(self, text, frame_size, fontsize=36, color=(255, 255, 255), position='bottom', subtitle_font=None):
        """创建包含文本的透明帧，支持中文和日文"""
        try:
            # 创建透明背景
            frame = np.zeros((frame_size[1], frame_size[0], 4), dtype=np.uint8)
            pil_img = Image.fromarray(frame)
            draw = ImageDraw.Draw(pil_img)

               # 检测操作系统类型
            is_windows = os.name == 'nt'
            
            # Windows 系统字体路径
            windows_font_paths = {
                'default': 'C:/Windows/Fonts/msgothic.ttc',  # 日文默认字体
                'watermark': 'C:/Windows/Fonts/msyh.ttc',    # 中文默认字体
                'japanese': {
                    1: 'C:/Windows/Fonts/msgothic.ttc',      # MS Gothic
                    2: 'C:/Windows/Fonts/msmincho.ttc',      # MS Mincho
                    3: 'C:/Windows/Fonts/YuGothM.ttc',       # Yu Gothic Medium
                    4: 'C:/Windows/Fonts/YuGothB.ttc',       # Yu Gothic Bold
                    5: 'C:/Windows/Fonts/msgothic.ttc',      # Yu Mincho Regular
                    6: 'C:/Windows/Fonts/msgothic.ttc',      # Yu Mincho Bold
                    7: 'C:/Windows/Fonts/msgothic.ttc',      # Yu Mincho Bold
                    8: 'C:/Windows/Fonts/msgothic.ttc',      # Yu Mincho Bold
                    9: 'C:/Windows/Fonts/msgothic.ttc',      # Yu Mincho Bold
                    10: 'C:/Windows/Fonts/msgothic.ttc',      # Yu Mincho Bold
                }
            }

            '''
            sudo apt-get update && sudo apt-get install -y \
                fonts-ipafont-gothic \
                fonts-ipafont-mincho \
                fonts-japanese-gothic \
                fonts-japanese-mincho \
                fonts-noto-cjk \
                fonts-noto-cjk-extra \
                fontconfig
            '''
            # Linux 系统字体路径
            linux_font_paths = {
                'watermark': f"{os.path.join(os.getcwd(), 'fontstyle', 'moon_get-Heavy.ttf')}",
                'default': "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # 添加 DejaVu 作为默认字体
                'japanese': {
                    1: "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",      # IPA Gothic
                    2: "/usr/share/fonts/opentype/ipafont-mincho/ipam.ttf",      # IPA Mincho
                    3: "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", # Noto Sans CJK
                    4: "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",    # Noto Sans CJK Bold
                    5: "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",# Noto Serif CJK
                    6: "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",   # Noto Serif CJK Bold
                    7: "/usr/share/fonts/truetype/takao-gothic/TakaoGothic.ttf", # Takao Gothic
                    8: "/usr/share/fonts/truetype/takao-mincho/TakaoMincho.ttf", # Takao Mincho
                    9: "/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc",            # WenQuanYi Zen Hei
                    10: "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc"        # WenQuanYi Micro Hei
                },
            }


            # 检测文本语言类型
            def detect_language(text):
                # 日文字符范围
                jp_chars = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]')
                # 英文字符范围
                en_chars = re.compile(r'[a-zA-Z]')
                # 中文字符范围
                cn_chars = re.compile(r'[\u4e00-\u9fff]')
                
                jp_count = len(jp_chars.findall(text))
                en_count = len(en_chars.findall(text))
                cn_count = len(cn_chars.findall(text))
                
                if jp_count > en_count and jp_count > cn_count:
                    return 'ja'
                elif cn_count > en_count:
                    return 'zh'
                else:
                    return 'en'

            '''
            sudo apt-get update
            sudo apt-get install -y fonts-ipafont fonts-ipafont-gothic fonts-ipafont-mincho   # IPA字体
            sudo apt-get install -y fonts-noto-cjk fonts-noto-cjk-extra                       # Google Noto CJK字体
            sudo apt-get install -y fonts-japanese-gothic fonts-japanese-mincho               # 日文Gothic和Mincho字体
            
            # 安装必要的日文字体包
            sudo apt-get update
            sudo apt-get install -y fonts-noto-cjk-extra fonts-japanese-gothic fonts-japanese-mincho fonts-noto-cjk fonts-ipafont-gothic fonts-ipafont-mincho fonts-ipafont fonts-vlgothic fonts-hanazono fonts-mplus
            sudo fc-cache -f -v


            # 复制自定义字体到系统目录
            sudo mkdir -p /usr/share/fonts/truetype/custom-fonts
            sudo cp fontstyle/ja/*.ttf /usr/share/fonts/truetype/custom-fonts/
            sudo chmod 755 /usr/share/fonts/truetype/custom-fonts
            sudo chmod 644 /usr/share/fonts/truetype/custom-fonts/*.ttf
            sudo chown -R root:root /usr/share/fonts/truetype/custom-fonts

   
            rm -rf fontstyle/ja/*.ttf
            mkdir -p fontstyle/ja
    
            for font in /usr/share/fonts/truetype/custom-fonts/*.ttf; do
                basename=$(basename "$font")
                sudo ln -s "$font" "fontstyle/ja/$basename"
            done
    
    
            sudo fc-cache -f -v

            '''
            # 根据操作系统选择字体路径
            subtitle_font_paths = windows_font_paths if is_windows else linux_font_paths
            
            # 加载字体
            font = None
            self.logger.info(f'====> 字体是 subtitle_font: {subtitle_font}')
            
            # 检测文本语言
            text_language = detect_language(text)
            self.logger.info(f'Detected language: {text_language}')
            
            # 根据语言和字体选择选择合适的字体
            if text_language == 'ja':
                if isinstance(subtitle_font, (int, float)) and 1 <= subtitle_font <= 8:
                    font_path = subtitle_font_paths['japanese'][subtitle_font]
                else:
                    font_path = subtitle_font_paths['japanese'][1]
            else:
                # 其他情况使用默认字体
                font_path = subtitle_font_paths['default']
            
            # 水印字体特殊处理
            if subtitle_font == 'watermark':
                if text_language == 'ja':
                    font_path = subtitle_font_paths['default']
                else:
                    font_path = subtitle_font_paths['watermark']
            
            # 尝试加载字体
            if os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(font_path, fontsize)
                    self.logger.info(f"Using font: {os.path.basename(font_path)}")
                except Exception as e:
                    self.logger.warning(f"Failed to load font {font_path}: {str(e)}")
                    # 加载失败时使用默认字体
                    try:
                        font = ImageFont.truetype(subtitle_font_paths['default'], fontsize)
                        self.logger.info("Using default fallback font")
                    except Exception as e:
                        self.logger.error(f"Failed to load default font: {str(e)}")
                        font = ImageFont.load_default()
            else:
                self.logger.error(f"Font file not found: {font_path}")
                font = ImageFont.load_default()

            
            # 测试字体是否能正确渲染文本
            try:
                # 获取文本大小
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                self.logger.info(f"Text size: {text_width}x{text_height}")
            except Exception as e:
                self.logger.error(f"Error measuring text: {str(e)}")
                return frame
            
            # 计算文本位置
            x, y = self._calculate_text_position(position, frame_size, text_width, text_height)
            
            # 添加文本描边（增强可读性）
            outline_color = (0, 0, 0, 255)
            outline_width = 2
            for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                for i in range(outline_width):
                    draw.text((x + dx + i, y + dy + i), text, font=font, fill=outline_color)
            
            # 添加主文本
            draw.text((x, y), text, font=font, fill=(*color, 255))
            
            return np.array(pil_img)
            
        except Exception as e:
            self.logger.error(f"Error creating text frame: {str(e)}")
            return np.zeros((frame_size[1], frame_size[0], 4), dtype=np.uint8)
        
    


    '''
    升级版本
    '''
    def merge_clips(self, clip_mapping, clips, merged_video_path):
        if not clip_mapping:
            raise ValueError("No clips were successfully downloaded")
        
        try:
            return self.concatenate_videos(clip_mapping, clips, merged_video_path)
        except Exception as e:
            self.logger.error(f"Error in merge_clips: {str(e)}")
            raise

    
    # 计算每行最大字符数（根据视频宽度和字体大小估算）
    def calculate_max_chars(self,text, frame_width, fontsize):
        """
            计算每行最大字符数
            
            Args:
                text (str): 要显示的文本
                frame_width (int): 视频帧宽度
                fontsize (int): 字体大小
                            
                Returns:
                int: 每行最大字符数
        """
        # 降低基础系数，允许更多字符
        base_factor = 0.8  # 从 1.2 降低到 0.8
                        
        # 调整字体大小系数，使其影响更小
        size_factor = 1.0 + (fontsize / 150)  # 从 100 增加到 150，减小字体大小的影响
                        
        # 计算中日韩字符的比例
        cjk_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff')
        total_chars = len(text)
        cjk_ratio = cjk_chars / total_chars if total_chars > 0 else 0
                        
        # 降低 CJK 字符的宽度系数
        width_factor = (cjk_ratio * 1.2) + ((1 - cjk_ratio) * 0.5)  # CJK 系数从 1.4 降到 1.2

        # 最终计算
        final_factor = base_factor * size_factor * width_factor
        max_chars = int(frame_width / (fontsize * final_factor))
                        
        # 增加最小值
        min_chars = max(8, int(frame_width / (fontsize * 2)))  # 确保至少有 8 个字符，或基于宽度的动态最小值
                        
        return max(max_chars, min_chars)
                    
    def split_text_into_lines(self,text, max_chars_per_line):
        """将文本分割成多行"""
        if len(text) <= max_chars_per_line:
            return [text]
                                
        lines = []
        current_line = ""
                            
        # 按标点符号和空格分割
        parts = re.findall(r'[^，。！？,.!?\s]+[，。！？,.!?\s]?', text)
                            
        for part in parts:
            if len(current_line + part) <= max_chars_per_line:
                current_line += part
            else:
                if current_line:
                    lines.append(current_line)
                    current_line = part
                                    
                # 如果单个部分超过最大长度，强制分割
                while len(current_line) > max_chars_per_line:
                    lines.append(current_line[:max_chars_per_line])
                    current_line = current_line[max_chars_per_line:]
                            
            if current_line:
                lines.append(current_line)
                                
        return lines
            
                    
    def process_frame(self, get_frame, t, preset, subtitle_segments):
        # 获取当前帧
        frame = get_frame(t)
        frame_width = frame.shape[1]
                        
        # 检查是否需要添加字幕
        if preset.get('subtitleFont', 0) == 0 or preset.get('subtitleStyle', 0) == 0:
            return frame
                        
        # 找到当前时间应该显示的句子
        current_subtitle = None
        for segment in subtitle_segments:
            if segment['start'] <= t <= segment['end']:
                current_subtitle = segment['text']
                break
                        
        if not current_subtitle:
            return frame
                        
        # 设置字幕样式
        fontsize = {
            0: 0,     # 不显示字幕
            1: 24,    # 小
            2: 48,    # 中
            3: 60     # 大
        }.get(preset.get('subtitleStyle', 2), 48)  # 默认使用中等大小
                        
        # 如果字体大小为0，不显示字幕
        if fontsize == 0:
            return frame
                        
        # 计算每行最大字符数（根据视频宽度和字体大小估算）
        def calculate_max_chars(text, frame_width, fontsize):
            """
            计算每行最大字符数
            
            Args:
                text (str): 要显示的文本
                frame_width (int): 视频帧宽度
                fontsize (int): 字体大小
                            
                Returns:
                    int: 每行最大字符数
                """
            # 降低基础系数，允许更多字符
            base_factor = 0.8  # 从 1.2 降低到 0.8
                        
            # 调整字体大小系数，使其影响更小
            size_factor = 1.0 + (fontsize / 150)  # 从 100 增加到 150，减小字体大小的影响
                        
            # 计算中日韩字符的比例
            cjk_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff')
            total_chars = len(text)
            cjk_ratio = cjk_chars / total_chars if total_chars > 0 else 0
                        
            # 降低 CJK 字符的宽度系数
            width_factor = (cjk_ratio * 1.2) + ((1 - cjk_ratio) * 0.5)  # CJK 系数从 1.4 降到 1.2
                        
            # 最终计算
            final_factor = base_factor * size_factor * width_factor
            max_chars = int(frame_width / (fontsize * final_factor))
                        
            # 增加最小值
            min_chars = max(8, int(frame_width / (fontsize * 2)))  # 确保至少有 8 个字符，或基于宽度的动态最小值
                        
            return max(max_chars, min_chars)
                    
        def split_text_into_lines(text, max_chars_per_line):
            """将文本分割成多行"""
            if len(text) <= max_chars_per_line:
                return [text]
                                
            lines = []
            current_line = ""
                            
            # 按标点符号和空格分割
            parts = re.findall(r'[^，。！？,.!?\s]+[，。！？,.!?\s]?', text)
                            
            for part in parts:
                if len(current_line + part) <= max_chars_per_line:
                    current_line += part
                else:
                    if current_line:
                        lines.append(current_line)
                        current_line = part
                                    
                        # 如果单个部分超过最大长度，强制分割
                        while len(current_line) > max_chars_per_line:
                            lines.append(current_line[:max_chars_per_line])
                            current_line = current_line[max_chars_per_line:]
                            
                if current_line:
                    lines.append(current_line)
                                
                return lines
        # 使用新的计算方法
        max_chars_per_line = calculate_max_chars(current_subtitle, frame_width, fontsize)
        # 将字幕文本分割成多行
        subtitle_lines = split_text_into_lines(current_subtitle, max_chars_per_line)
                        
        # 将多行文本合并，用换行符连接
        formatted_subtitle = '\n'.join(subtitle_lines)

        # 将16进制颜色转换为RGB
        try:
            hex_color = preset.get('subtitleColor', '#FFFFFF').lstrip('#')
            font_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        except Exception as e:
            self.logger.warning(f"Error parsing subtitle color: {str(e)}, using default white")
            font_color = (255, 255, 255)
                        
        # 创建当前字幕的帧
        self.logger.info(f'===> 字幕: {formatted_subtitle}')

        subtitle_frame = self.create_text_frame(
                        formatted_subtitle,
                        frame.shape[:2][::-1],
                        fontsize=fontsize,
                        color=font_color,
                        position='bottom',
                        subtitle_font=preset.get('subtitleFont', 1)
                    )


        # 转换帧为RGBA
        frame_rgba = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
                        
        # 计算字幕的 alpha 遮罩
        alpha_subtitle = subtitle_frame[:, :, 3:] / 255.0
        alpha_frame = 1.0 - alpha_subtitle
                        
        # 添加半透明黑色背景
        bg_color = np.zeros_like(frame_rgba)
        bg_color[:, :, 3] = alpha_subtitle[:, :, 0] * 128
                        
        # 混合原始帧、背景和字幕
        for c in range(3):
            frame_rgba[:, :, c] = (
                    frame_rgba[:, :, c] * alpha_frame[:, :, 0] +
                    bg_color[:, :, c] * (alpha_subtitle[:, :, 0] * 0.5) +
                    subtitle_frame[:, :, c] * alpha_subtitle[:, :, 0]
                )
                        
        return cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2RGB)
        


    def _parse_color(self, hex_color):
        """解析16进制颜色值"""
        try:
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        except Exception as e:
            self.logger.warning(f"Error parsing color: {str(e)}, using default white")
            return (255, 255, 255)

    def _compose_frame(self, frame, subtitle_frame):
        """合成带字幕的帧"""
        frame_rgba = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
        
        # 计算alpha遮罩
        alpha_subtitle = subtitle_frame[:, :, 3:] / 255.0
        alpha_frame = 1.0 - alpha_subtitle
        
        # 添加半透明黑色背景
        bg_color = np.zeros_like(frame_rgba)
        bg_color[:, :, 3] = alpha_subtitle[:, :, 0] * 128
        
        # 混合图层
        for c in range(3):
            frame_rgba[:, :, c] = (
                frame_rgba[:, :, c] * alpha_frame[:, :, 0] +
                bg_color[:, :, c] * (alpha_subtitle[:, :, 0] * 0.5) +
                subtitle_frame[:, :, c] * alpha_subtitle[:, :, 0]
            )
        
        return cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2RGB)


    def combine_video_audio_subtitle(
        self,
        video_paths: list,
        audio_path: str,
        subtitle_path: str,
        font_path: str,
        output_path: str,
        subtitle_style: dict = None
    ):
        """合成多个视频、音频和字幕"""
        temp_files = []  # 用于存储临时文件路径
        
        try:
            # 首先合并所有视频
            self.logger.info("开始处理多个视频文件...")
            combined_video, temp_video_files = self.combine_videos(video_paths)
            temp_files.extend(temp_video_files)
            temp_files.append(combined_video)
            
            # 获取合并后视频的信息
            probe = ffmpeg.probe(combined_video)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            width = int(video_info['width'])
            height = int(video_info['height'])
            self.logger.info(f"合并后视频分辨率: {width}x{height}")
            
            # 获取视频和音频时长
            video_duration = self.get_media_duration(combined_video)
            audio_duration = self.get_media_duration(audio_path)
            max_duration = max(video_duration, audio_duration)
            
            # 默认字幕样式
            default_style = {
                'FontName': os.path.splitext(os.path.basename(font_path))[0],
                'FontSize': 24,
                'PrimaryColour': '&HFFFFFF',
                'OutlineColour': '&H000000',
                'Bold': 1,
                'Outline': 2,
            }
            if subtitle_style:
                default_style.update(subtitle_style)
            
            style_str = ','.join([f"{k}={v}" for k, v in default_style.items()])
            
            # 处理视频流
            video = ffmpeg.input(combined_video)
            if video_duration < max_duration:
                black_bg = ffmpeg.input(f'color=c=black:s={width}x{height}:d={max_duration}', f='lavfi')
                video = ffmpeg.filter(
                    [black_bg, video],
                    'overlay',
                    shortest=0,
                    repeatlast=1
                )
            
            # 处理音频流
            audio = ffmpeg.input(audio_path)
            if audio_duration < max_duration:
                silence = ffmpeg.input(f'anullsrc=d={max_duration}', f='lavfi')
                audio = ffmpeg.filter(
                    [silence, audio],
                    'amix',
                    duration='first'
                )
            
            # 添加字幕
            video_with_subtitle = video.filter('subtitles', subtitle_path, 
                force_style=style_str)
            
            # 合并所有流
            output = ffmpeg.output(
                video_with_subtitle,
                audio,
                output_path,
                acodec='aac',
                vcodec='h264',
                video_bitrate='2000k',
                audio_bitrate='192k',
                pix_fmt='yuv420p',
                movflags='+faststart',
                strict='experimental'
            ).overwrite_output()
            
            # 打印 FFmpeg 命令
            self.logger.info("执行的FFmpeg命令:")
            self.logger.info(" ".join(ffmpeg.get_args(output)))
            
            # 运行命令
            ffmpeg.run(output)
            self.logger.info(f"视频合成成功！输出文件：{output_path}")
            self.logger.info(f"最终视频时长：{max_duration:.2f}秒")
            
        except ffmpeg.Error as e:
            error_message = e.stderr.decode() if e.stderr is not None else str(e)
            self.logger.error(f"发生错误：{error_message}")
        finally:
            # 清理临时文件
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        self.logger.info(f"已清理临时文件：{temp_file}")
                except Exception as e:
                    self.logger.error(f"清理临时文件失败：{e}")
    




    def _get_adjustment_prompt(self, current_chars, target_chars, content, lang):
        """获取调整提示词"""
        prompts = {
            'en': f"""Rewrite the following narration in an engaging social media style:
                Original ({current_chars} chars): {content}
                Rules:
                1. Use trendy, conversational language that resonates with young audiences
                2. Create a personal, authentic tone
                3. Balance informative content with emotional appeal
                4. STRICTLY keep length to {target_chars} characters (±5%)
                5. Make content relatable and shareable
                6. Consider natural speaking pace (about 2-3 chars per second)
                7. Add hooks and engaging transitions
                8. NO emojis or special characters allowed
                Return only the rewritten script without any explanations.""",
            
            'ja': f"""以下のナレーションを、SNSで人気の魅力的な語り口で書き直してください：
                原文（{current_chars}文字）: {content}
                ルール：
                1. 若者に響く、トレンド感のある表現を使用
                2. 親近感のある、自然な話し方
                3. 共感を呼ぶストーリー展開
                4. 厳密に{target_chars}文字（±5%）に調整
                5. 視聴者の興味を引く魅力的な表現
                6. 自然な話速を考慮（1秒あたり2-3文字程度）
                7. 印象的な導入と展開を心がける
                8. 絵文字や特殊文字は使用禁止
                脚本形式のテキストのみを返してください。""",
            
            'zh': f"""将以下旁白改写成小红书风格的吸引人的语言：
                原文（{current_chars}字符）: {content}
                规则：
                1. 使用年轻人喜欢的网感表达
                2. 保持亲和力和真实感
                3. 用故事化表达增强共鸣感
                4. 严格控制在{target_chars}字符（±5%）
                5. 运用吸引眼球的开场和过渡
                6. 考虑自然语速（每秒2-3个字符）
                7. 增加悬念和引导性表达
                8. 禁止使用表情符号和特殊字符
                仅返回脚本化的旁白文本，无需解释。"""
        }
        return prompts.get(lang, prompts['ja'])
    

    def _apply_subtitles_to_video(self, video, subtitle_segments, preset):
        """将字幕应用到视频上,与音频旁白同步的打轴效果"""
        
        if video is None:
            self.logger.error("Input video is None")
            return None

        test_frame = video.get_frame(0)
        if test_frame is None:
            self.logger.error("Cannot get first frame from video")
            return None

        self.logger.info(f"Video size: {video.size}, duration: {video.duration}")
        
        # 添加日志输出检查字幕段落
        self.logger.info(f"Total subtitle segments: {subtitle_segments}")
        for seg in subtitle_segments:
            self.logger.info(f"Subtitle segment: {seg['start']:.2f}s - {seg['end']:.2f}s: {seg['text']}")

        # 预处理字幕样式
        subtitle_style = {
            'fontsize': {0: 0, 1: 24, 2: 48, 3: 60}.get(
                preset.get('subtitleStyle', 2), 48
            ),
            'font': preset.get('subtitleFont', 1),
            'color': self._parse_color(preset.get('subtitleColor', '#FFFFFF'))
        }

        def find_current_subtitle(t):
            """查找当前时间点对应的字幕内容"""
            t = round(t, 3)
            
            # 直接遍历查找当前时间点的字幕
            current_segment = None
            for segment in subtitle_segments:
                if segment['start'] <= t <= segment['end']:
                    current_segment = segment
                    break
                    
            if current_segment:
                text = current_segment['text']
                
                # 计算每行最大字符数
                max_chars_per_line = max(20, int(video.w / (subtitle_style['fontsize'] * 1.5)))
                
                # 分行处理
                lines = []
                current_line = ""
                
                # 按字符处理，确保不会在字符中间断行
                for char in text:
                    if len(current_line + char) <= max_chars_per_line:
                        current_line += char
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = char
                
                # 添加最后一行
                if current_line:
                    lines.append(current_line)
                
                return '\n'.join(lines)
                    
            return None

        def process_frame_with_subtitle(gf, t):
            """处理单个帧的字幕"""
            try:
                # 获取当前帧
                frame = gf(t)
                
                # 获取当前时间的字幕
                current_subtitle = find_current_subtitle(t)
                if not current_subtitle:
                    return frame
                
                # 创建字幕帧
                subtitle_frame = self.create_text_frame(
                    current_subtitle,
                    frame.shape[:2][::-1],
                    fontsize=subtitle_style['fontsize'],
                    color=subtitle_style['color'],
                    position='bottom',
                    subtitle_font=subtitle_style['font']
                )
                
                # 合成帧
                return self._compose_frame(frame, subtitle_frame)
                
            except Exception as e:
                self.logger.error(f"Error processing frame at {t}s: {str(e)}")
                return frame

        # 应用字幕处理
        self.logger.info('Applying subtitles to video...')
        try:
            return video.fl(process_frame_with_subtitle)
        except Exception as e:
            self.logger.error(f"Error applying subtitles to video: {str(e)}")
            return video

    def _compose_frame(self, frame, subtitle_frame):
        """合成带字幕的帧"""
        frame_rgba = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
        
        # 计算alpha遮罩
        alpha_subtitle = subtitle_frame[:, :, 3:] / 255.0
        alpha_frame = 1.0 - alpha_subtitle
        
        # 添加半透明黑色背景
        bg_color = np.zeros_like(frame_rgba)
        bg_color[:, :, 3] = alpha_subtitle[:, :, 0] * 128
        
        # 混合图层
        for c in range(3):
            frame_rgba[:, :, c] = (
                frame_rgba[:, :, c] * alpha_frame[:, :, 0] +
                bg_color[:, :, c] * (alpha_subtitle[:, :, 0] * 0.5) +
                subtitle_frame[:, :, c] * alpha_subtitle[:, :, 0]
            )
        
        return cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2RGB)

    def _compress_narration_content(self, content, compression_ratio, lang='zh'):
        """
        使用 LLM 压缩旁白内容
        
        Args:
            content (str): 原始内容
            compression_ratio (float): 压缩比例 (0-1)
            lang (str): 语言代码
        
        Returns:
            str: 压缩后的内容
        """
        prompts = {
            'zh': f"""请将以下文本压缩到原有长度的{int(compression_ratio * 100)}%，保持核心信息和语言流畅性：
                原文：{content}
                要求：
                1. 保持语言自然流畅
                2. 保留最重要的信息
                3. 确保句子完整
                4. 维持叙述语气
                5. 避免重复内容
                仅返回压缩后的文本，无需解释。""",
                
            'en': f"""Compress the following text to {int(compression_ratio * 100)}% of its original length while maintaining core information and fluency:
                Original: {content}
                Requirements:
                1. Keep language natural and flowing
                2. Retain most important information
                3. Ensure complete sentences
                4. Maintain narrative tone
                5. Avoid repetition
                Return only the compressed text without explanation.""",
                
            'ja': f"""以下のテキストを元の長さの{int(compression_ratio * 100)}%に圧縮し、核心的な情報と言葉の流れを保持してください：
                原文：{content}
                要件：
                1. 自然な言葉の流れを保つ
                2. 最も重要な情報を残す
                3. 文章の完全性を確保
                4. ナレーション調を維持
                5. 重複を避ける
                圧縮されたテキストのみを返し、説明は不要です。"""
        }
        
        try:
            prompt = prompts.get(lang, prompts['ja'])
            compressed = self.remote_llm_util.native_chat(prompt)
            return compressed.strip()
        except Exception as e:
            self.logger.error(f"Error compressing narration content: {e}")
            raise

    def _process_video_with_effects(self, input_video_path, project_no, subtitle_path=None, watermark=None, bgm_path=None, narration_path=None):
        """处理视频特效、音频混合和字幕的详细实现"""
        try:
            # 准备输出路径
            output_dir = os.path.join(self.project_dir, 'post_process', project_no)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{project_no}_processed.mp4")

            # 构建基本的过滤器链
            filter_chains = []
            
            # 视频过滤器
            if subtitle_path:
                filter_chains.append(f"subtitles={subtitle_path}")
            if watermark:
                filter_chains.append(
                    f"drawtext=text='{watermark}':fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
                    ":fontsize=20:fontcolor=white:x=10:y=10"
                )

            # 音频过滤器
            audio_inputs = []
            if narration_path:
                audio_inputs.append(f"[1:a]volume=1.0[narration]")
            if bgm_path:
                # 使用采样数而不是时间来设置 aloop
                audio_inputs.append(f"[2:a]volume=0.3,aloop=loop=-1:size=44100[bgm]")
            
            # 合并音频
            if audio_inputs:
                if len(audio_inputs) == 1:
                    audio_mix = audio_inputs[0]
                else:
                    audio_mix = f"{';'.join(audio_inputs)};[narration][bgm]amix=inputs=2:duration=first[aout]"
            else:
                audio_mix = "[0:a]acopy[aout]"

            # 合并所有过滤器
            filter_complex = ';'.join(filter_chains + [audio_mix])

            # 构建 FFmpeg 命令
            command = [
                'ffmpeg', '-y',
                '-i', input_video_path
            ]

            # 添加额外的输入
            if narration_path:
                command.extend(['-i', narration_path])
            if bgm_path:
                command.extend(['-i', bgm_path])

            # 添加过滤器和输出选项
            command.extend([
                '-filter_complex', filter_complex,
                '-map', '[0:v]',
                '-map', '[aout]',
                '-c:v', 'h264_nvenc',
                '-preset', 'p4',
                '-b:v', '5M',
                '-maxrate', '10M',
                '-bufsize', '10M',
                '-c:a', 'aac',
                '-b:a', '192k',
                output_path
            ])

            # 执行命令
            self.logger.info(f"Executing FFmpeg command: {' '.join(command)}")
            result = subprocess.run(
                command,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                self.logger.error(f"FFmpeg error: {result.stderr}")
                raise RuntimeError(f"FFmpeg processing failed: {result.stderr}")

            # 验证输出文件
            if not os.path.exists(output_path):
                raise FileNotFoundError(f"Output file was not created: {output_path}")

            file_size = os.path.getsize(output_path)
            if file_size == 0:
                raise ValueError(f"Output file is empty: {output_path}")

            return {
                'processed_video': output_path,
                'temp_dir': output_dir
            }

        except Exception as e:
            self.logger.error(f"Error in video processing: {str(e)}")
            return None
    

    def _get_audio_segments(self, audio_path: str,preset) -> List[dict]:
        """
        使用语音识别获取音频的时间戳信息
        """
        try:
            # 使用 whisper 或其他语音识别工具获取时间戳
            # 这里使用示例格式，需要根据实际使用的语音识别工具调整
            lang = preset.get('narrationLang', 'ja')
            
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

    def _align_subtitles_with_audio(
        self,
        narration_content: str,
        audio_segments: List[dict],
        audio_duration: float
    ) -> List[dict]:
        """
        将字幕文本与音频时间戳对齐
        """
        try:
            # 分割旁白文本为句子
            sentences = self._split_into_sentences(narration_content)
            
            # 如果音频段落数量与句子数量不匹配，进行调整
            if len(audio_segments) != len(sentences):
                # 使用文本相似度匹配
                aligned_segments = self._match_text_with_segments(
                    sentences,
                    audio_segments
                )
            else:
                # 直接使用音频段落的时间戳
                aligned_segments = [
                    {
                        'text': sentence,
                        'start': segment['start'],
                        'end': segment['end']
                    }
                    for sentence, segment in zip(sentences, audio_segments)
                ]
            
            return aligned_segments
            
        except Exception as e:
            self.logger.error(f"Error aligning subtitles: {str(e)}")
            return self._generate_subtitle_segments(narration_content, audio_duration)

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        将文本分割为句子
        """
        # 根据常见的句子结束符分割
        pattern = r'([^。！？.!?]+[。！？.!?])'
        sentences = []
        
        matches = re.finditer(pattern, text)
        for match in matches:
            sentence = match.group(1).strip()
            if sentence:
                sentences.append(sentence)
        
        # 处理最后一个可能没有标点的句子
        last_part = re.sub(pattern, '', text).strip()
        if last_part:
            sentences.append(last_part)
        
        return sentences

    def _match_text_with_segments(
        self,
        sentences: List[str],
        audio_segments: List[dict]
    ) -> List[dict]:
        """
        使用文本相似度匹配句子和音频段落
        """
        from difflib import SequenceMatcher
        
        aligned_segments = []
        used_segments = set()
        
        for sentence in sentences:
            best_match = None
            best_ratio = 0
            
            # 找到最匹配的音频段落
            for i, segment in enumerate(audio_segments):
                if i in used_segments:
                    continue
                    
                ratio = SequenceMatcher(
                    None,
                    sentence,
                    segment['text']
                ).ratio()
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = segment
            
            if best_match:
                used_segments.add(audio_segments.index(best_match))
                aligned_segments.append({
                    'text': sentence,
                    'start': best_match['start'],
                    'end': best_match['end']
                })
        
        return aligned_segments


    def _install_noto_fonts(self):
        """Install Noto Sans CJK JP fonts if not already installed"""
        try:
            # 首先检查字体是否已安装
            fc_list = subprocess.run(['fc-list', ':', 'family'], capture_output=True, text=True)
            if 'Noto Sans CJK JP' in fc_list.stdout:
                self.logger.info("Noto Sans CJK JP already installed")
                return True
                
            # 检查常见的字体文件位置
            font_dirs = [
                '/usr/share/fonts/opentype/noto',
                '/usr/share/fonts/truetype/noto',
                '/usr/local/share/fonts',
                os.path.expanduser('~/.fonts'),
                os.path.expanduser('~/.local/share/fonts')
            ]
            
            font_patterns = [
                '*Noto*CJK*JP*.otf',
                '*Noto*CJK*JP*.ttf',
                '*noto*cjk*jp*.otf',
                '*noto*cjk*jp*.ttf'
            ]
            
            # 检查所有可能的目录
            for font_dir in font_dirs:
                if not os.path.exists(font_dir):
                    continue
                    
                for pattern in font_patterns:
                    matches = glob.glob(os.path.join(font_dir, '**', pattern), recursive=True)
                    if matches:
                        self.logger.info(f"Found Noto fonts at: {matches[0]}")
                        # 更新字体缓存以确保系统识别
                        subprocess.run(['fc-cache', '-f', '-v'], check=True)
                        return True
            
            # 如果没有找到字体，则进行安装
            self.logger.info("Noto fonts not found, installing...")
            
            # 检查是否为 root 用户或有 sudo 权限
            has_sudo = subprocess.run(['sudo', '-n', 'true'], capture_output=True).returncode == 0
            
            if has_sudo:
                # 使用 apt 安装字体
                subprocess.run(['sudo', 'apt-get', 'update'], check=True)
                subprocess.run(['sudo', 'apt-get', 'install', '-y', 'fonts-noto-cjk'], check=True)
                subprocess.run(['sudo', 'fc-cache', '-f', '-v'], check=True)
            else:
                # 如果没有 sudo 权限，尝试用户级安装
                user_font_dir = os.path.expanduser('~/.local/share/fonts')
                os.makedirs(user_font_dir, exist_ok=True)
                
                # 这里可以添加从可信源下载字体的逻辑
                self.logger.warning("No sudo rights, please install Noto fonts manually or provide sudo access")
                return False
            
            # 验证安装
            fc_list = subprocess.run(['fc-list', ':', 'family'], capture_output=True, text=True)
            if 'Noto Sans CJK JP' in fc_list.stdout:
                self.logger.info("Noto fonts successfully installed")
                return True
            else:
                self.logger.error("Font installation verification failed")
                return False
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {e.cmd}, Return code: {e.returncode}")
            self.logger.error(f"Error output: {e.stderr}")
            return False
        except Exception as e:
            self.logger.error(f"Error installing Noto fonts: {str(e)}")
            return False
    
    def _process_subtitle_linebreaks(self, subtitle_path, chars_per_line, fontsize):
        """处理字幕文件，添加适当的换行"""
        try:
            processed_path = subtitle_path.replace('.srt', '_processed.srt')
            
            with open(subtitle_path, 'r', encoding='utf-8') as f_in, \
                 open(processed_path, 'w', encoding='utf-8') as f_out:
                
                content = f_in.read()
                blocks = content.strip().split('\n\n')
                
                for block in blocks:
                    lines = block.split('\n')
                    if len(lines) >= 3:
                        f_out.write(f"{lines[0]}\n")  # 序号
                        f_out.write(f"{lines[1]}\n")  # 时间码
                        
                        # 处理文本，考虑字体大小调整每行长度
                        text = ' '.join(lines[2:])
                        # 较大字体时减少每行字符数
                        adjusted_chars = int(chars_per_line * (1 - (fontsize - 8) * 0.05)) if fontsize > 8 else chars_per_line
                        wrapped_text = self._wrap_text(text, adjusted_chars)
                        f_out.write(wrapped_text + '\n\n')
            
            return processed_path
            
        except Exception as e:
            self.logger.error(f"Error processing subtitle linebreaks: {str(e)}")
            return subtitle_path
            
    def _wrap_text(self, text, chars_per_line):
        """智能换行文本"""
        if not text:
            return text
            
        # 检测文本类型（中文/英文）
        is_cjk = any('\u4e00' <= char <= '\u9fff' for char in text)
        
        if is_cjk:
            return self._wrap_cjk_text(text, chars_per_line)
        else:
            return self._wrap_latin_text(text, chars_per_line)
            
    def _wrap_cjk_text(self, text, chars_per_line):
        """处理中日韩文本的换行"""
        lines = []
        current_line = ''
        
        # 标点符号列表
        punctuation = '，。！？、：；,.!?:;'
        
        for char in text:
            if len(current_line) >= chars_per_line:
                # 查找最近的标点符号位置
                last_punct = -1
                for i in range(len(current_line) - 1, max(len(current_line) - 5, -1), -1):
                    if current_line[i] in punctuation:
                        last_punct = i
                        break
                
                if last_punct != -1 and len(current_line) - last_punct <= 5:
                    # 在标点处换行
                    lines.append(current_line[:last_punct + 1])
                    current_line = current_line[last_punct + 1:] + char
                else:
                    # 直接换行
                    lines.append(current_line)
                    current_line = char
            else:
                current_line += char
                
        if current_line:
            lines.append(current_line)
            
        # 限制最大行数
        if len(lines) > 3:
            # 尝试重新组合行
            new_lines = []
            current_line = ''
            for line in lines:
                if len(current_line) + len(line) <= chars_per_line * 1.2:  # 允许稍微超过一点
                    current_line = (current_line + line) if current_line else line
                else:
                    if current_line:
                        new_lines.append(current_line)
                    current_line = line
            if current_line:
                new_lines.append(current_line)
            lines = new_lines[:3]  # 限制为最多3行
            
        return '\n'.join(lines)


    def _wrap_latin_text(self, text, chars_per_line):
        """处理英文等拉丁文字的换行"""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word)
            # 计算添加这个词后的长度（包括空格）
            new_length = current_length + word_length + (1 if current_line else 0)
            
            if new_length <= chars_per_line:
                # 当前行还能容纳这个词
                current_line.append(word)
                current_length = new_length
            else:
                # 需要换行
                if current_line:
                    lines.append(' '.join(current_line))
                
                # 如果单词本身就超过每行长度，需要强制断词
                if word_length > chars_per_line:
                    # 按照每行长度切分长单词
                    while word:
                        lines.append(word[:chars_per_line])
                        word = word[chars_per_line:]
                    current_line = []
                    current_length = 0
                else:
                    # 新起一行
                    current_line = [word]
                    current_length = word_length
        
        # 添加最后一行
        if current_line:
            lines.append(' '.join(current_line))
            
        # 限制最大行数为3行
        if len(lines) > 3:
            # 尝试重新组合行
            combined_lines = []
            i = 0
            while i < len(lines):
                if i + 1 < len(lines):
                    # 尝试合并相邻的两行
                    combined = lines[i] + ' ' + lines[i + 1]
                    if len(combined) <= chars_per_line * 1.2:  # 允许稍微超过一点
                        combined_lines.append(combined)
                        i += 2
                        continue
                combined_lines.append(lines[i])
                i += 1
            
            # 如果还是超过3行，只保留前3行
            lines = combined_lines[:3]
        
        return '\n'.join(lines)


    def get_video_duration(self,video_path):
        # 1. 验证输入
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")
                
        # 2. 获取视频信息
        probe = ffmpeg.probe(video_path)
        video_info = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        if not video_info:
            raise ValueError("No video stream found in input file")
        video_duration = float(probe['format']['duration'])
        return video_duration
    



    def apply_effects(self, video_path: str, subtitle_path: str, narration_audio_path: str, preset: dict, project_no: str) -> Tuple[Optional[str], Optional[str]]:
        """
        应用视频特效，包括旁白、字幕、水印和背景音乐
        使用 ffmpeg-python 处理
        """
        try:
            # 1. 验证输入
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Input video not found: {video_path}")
                
            # 2. 获取视频信息
            probe = ffmpeg.probe(video_path)
            video_info = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            if not video_info:
                raise ValueError("No video stream found in input file")
            video_duration = float(probe['format']['duration'])
            
            # 3. 准备输出路径
            temp_dir = Path('temp') / project_no
            temp_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(temp_dir / f"processed_{Path(video_path).stem}.mp4")

            # 4. 构建处理流程
            stream = ffmpeg.input(video_path)
            
            # 5. 处理字幕
            if preset.get('subtitle') and subtitle_path and os.path.exists(subtitle_path):
                # 字幕样式设置
                fontsize = min(max(preset.get('subtitleStyle', 1), 1), 10) * 1.5 + 16
                color = preset.get('subtitleColor', '#FFFFFF').lstrip('#')
                color_bgr = color[4:6] + color[2:4] + color[0:2]
                
                # 添加字幕滤镜
                stream = stream.filter('subtitles', subtitle_path, 
                    force_style=f'FontName=Noto Sans CJK JP,'
                            f'FontSize={fontsize},'
                            f'PrimaryColour=&H{color_bgr}&,'
                            f'Alignment=2,'
                            f'BorderStyle=1,'
                            f'Outline=1,'
                            f'Shadow=1,'
                            f'MarginV=20')

            # 6. 处理水印
            if preset.get('watermark'):
                watermark_text = preset.get('watermarkText', '')
                if watermark_text:
                    position = preset.get('watermarkPosition', 0)
                    position_map = {
                        0: "(w-text_w)/2:h-th-10",  # 底部居中
                        1: "10:10",                 # 左上
                        2: "w-tw-10:10",           # 右上
                        3: "(w-text_w)/2:(h-th)/2", # 中间
                        4: "10:h-th-10",           # 左下
                        5: "w-tw-10:h-th-10"       # 右下
                    }
                    pos = position_map.get(position, position_map[0])
                    stream = stream.filter('drawtext', 
                        text=watermark_text,
                        fontsize=36,
                        fontcolor='white@0.3',
                        x=pos.split(':')[0],
                        y=pos.split(':')[1])

            # 7. 准备音频流
            audio_streams = []
            
            # 主视频音频
            main_audio = ffmpeg.input(video_path).audio
            audio_streams.append(main_audio)
            
            # 旁白音频
            if narration_audio_path and os.path.exists(narration_audio_path):
                narration = ffmpeg.input(narration_audio_path).audio.filter('volume', 1.0)
                audio_streams.append(narration)
            
            # 背景音乐
            if preset.get('bgm'):
                bgm_path = self._get_bgm_path(preset['bgm'])
                if bgm_path and os.path.exists(bgm_path):
                    bgm = ffmpeg.input(bgm_path).audio.filter('volume', 0.3)
                    audio_streams.append(bgm)

            # 8. 混合音频
            if len(audio_streams) > 1:
                audio = ffmpeg.filter(audio_streams, 'amix', inputs=len(audio_streams))
            else:
                audio = audio_streams[0]

            # 9. 输出设置
            try:
                # 尝试使用 GPU 编码
                output_stream = ffmpeg.output(
                    stream,
                    audio,
                    output_path,
                    **{
                        'c:v': 'h264_nvenc',    # GPU 编码
                        'preset': 'p4',          # 平衡速度和质量
                        'tune': 'hq',            # 高质量模式
                        'rc': 'vbr',             # 可变比特率
                        'cq': '23',              # 质量参数
                        'b:v': '5M',             # 视频比特率
                        'maxrate': '10M',        # 最大比特率
                        'bufsize': '10M',        # 缓冲区大小
                        'c:a': 'aac',            # 音频编码器
                        'b:a': '192k',           # 音频比特率
                        'ar': '48000',           # 音频采样率
                        'movflags': '+faststart' # Web优化
                    }
                ).overwrite_output()
                
                # 执行命令
                self.logger.info("Starting video processing with GPU acceleration...")
                ffmpeg.run(output_stream, capture_stdout=True, capture_stderr=True)
                
            except ffmpeg.Error as e:
                self.logger.warning("GPU processing failed, falling back to CPU...")
                # 如果 GPU 处理失败，使用 CPU 编码
                output_stream = ffmpeg.output(
                    stream,
                    audio,
                    output_path,
                    **{
                        'c:v': 'libx264',     # CPU 编码
                        'preset': 'medium',    # 平衡速度和质量
                        'crf': '23',          # 质量参数
                        'c:a': 'aac',
                        'b:a': '192k',
                        'ar': '48000',
                        'movflags': '+faststart'
                    }
                ).overwrite_output()
                
                ffmpeg.run(output_stream, capture_stdout=True, capture_stderr=True)

            # 10. 验证输出
            if not os.path.exists(output_path):
                raise FileNotFoundError(f"Output file not created: {output_path}")
                
            if os.path.getsize(output_path) == 0:
                raise ValueError(f"Output file is empty: {output_path}")

            self.logger.info(f"Successfully processed video: {output_path}")
            return output_path, video_duration

        except Exception as e:
            self.logger.error(f"Error in video processing: {str(e)}")
            if isinstance(e, ffmpeg.Error):
                self.logger.error(f"FFmpeg error: {e.stderr.decode() if e.stderr else 'Unknown error'}")
            return None, None

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


    def apply_effects_with_transition_by_ffmpeg(self, video_path: str, subtitle_path: str, narration_audio_path: str, preset: dict, project_no: str) -> Tuple[Optional[str], Optional[str]]:
        """应用视频特效，包括旁白、字幕、水印和背景音乐"""
        try:
            # 获取视频信息
            probe = ffmpeg.probe(video_path)
            video_duration = float(probe['format']['duration'])
            self.logger.info(f"Video duration: {video_duration:.2f} seconds")

            # 初始化输出路径
            temp_dir = Path('temp') / project_no
            temp_dir.mkdir(parents=True, exist_ok=True)
            final_output = str(temp_dir / f"final_{uuid.uuid4()}.mp4")
            
            try:
                # 构建 FFmpeg 命令
                command = ['ffmpeg', '-y']
                
                # 添加输入文件
                command.extend(['-i', video_path])
                if narration_audio_path and os.path.exists(narration_audio_path):
                    command.extend(['-i', narration_audio_path])
                
                # 添加背景音乐
                bgm_added = False
                if preset.get('bgm'):
                    bgm_path = self._get_bgm_path(preset['bgm'])
                    if bgm_path and os.path.exists(bgm_path):
                        command.extend(['-i', bgm_path])
                        bgm_added = True
                
                # 构建滤镜链
                filter_chains = []
                
                # 字幕滤镜
                if preset.get('subtitle') and subtitle_path and os.path.exists(subtitle_path):
                    self.logger.info("📝 Adding subtitles...")
                    fontsize = min(max(preset.get('subtitleStyle', 1), 1), 10) * 1.0 + 16
                    color = preset.get('subtitleColor', '#FFFFFF').lstrip('#')
                    color_bgr = color[4:6] + color[2:4] + color[0:2]
                    
                    subtitle_filter = f"subtitles='{subtitle_path}':force_style='FontName=Arial,FontSize={fontsize}," \
                                    f"PrimaryColour=&H{color_bgr}&,Alignment=2,BorderStyle=1,Outline=1,Shadow=1,MarginV=20'"
                    filter_chains.append(subtitle_filter)
                
                # 水印滤镜
                if preset.get('watermark'):
                    self.logger.info("💧 Adding watermark...")
                    watermark_text = preset.get('watermarkText', '')
                    position = preset.get('watermarkPosition', 0)
                    position_map = {
                        0: "(w-text_w)/2:h-th-10",  # 底部居中
                        1: "10:10",                 # 左上
                        2: "w-tw-10:10",           # 右上
                        3: "(w-text_w)/2:(h-th)/2", # 中间
                        4: "10:h-th-10",           # 左下
                        5: "w-tw-10:h-th-10"       # 右下
                    }
                    pos = position_map.get(position, position_map[0])
                    watermark_filter = f"drawtext=text='{watermark_text}':fontsize=72:fontcolor=white@0.3:" \
                                    f"x={pos.split(':')[0]}:y={pos.split(':')[1]}:box=0"
                    filter_chains.append(watermark_filter)
                
                # 音频滤镜
                self.logger.info("🎵 Processing audio streams...")
                
                # 处理音频
                if narration_audio_path and os.path.exists(narration_audio_path):
                    self.logger.info("🎙️ Adding narration audio...")
                    # 分步处理音频
                    temp_audio = str(temp_dir / f"temp_audio_{uuid.uuid4()}.m4a")  # 改用 m4a 格式
                    
                    # 先检查视频是否包含音频流
                    video_probe = ffmpeg.probe(video_path)
                    has_audio = any(stream['codec_type'] == 'audio' for stream in video_probe['streams'])
                    
                    # 构建音频混合命令
                    audio_cmd = ['ffmpeg', '-y']
                    
                    # 如果视频有音频流，添加视频输入
                    if has_audio:
                        audio_cmd.extend(['-i', video_path])
                    
                    # 添加旁白音频
                    audio_cmd.extend(['-i', narration_audio_path])
                    
                    # 添加背景音乐
                    if bgm_added:
                        audio_cmd.extend(['-i', bgm_path])


                    # 构建音频滤镜
                    filter_complex = []
                    if has_audio:
                        filter_complex.append('[0:a]volume=0.3[a0]')
                        filter_complex.append('[1:a]volume=1.0[a1]')
                        if bgm_added:
                            filter_complex.append('[2:a]volume=0.3,aloop=loop=-1:size=88200[a2]')
                            filter_complex.append('[a0][a1][a2]amix=inputs=3:duration=first:dropout_transition=0[aout]')
                        else:
                            filter_complex.append('[a0][a1]amix=inputs=2:duration=first:dropout_transition=0[aout]')
                    else:
                        # 如果视频没有音频流
                        filter_complex.append('[0:a]volume=1.0[a1]')
                        if bgm_added:
                            filter_complex.append('[1:a]volume=0.3,aloop=loop=-1:size=88200[a2]')
                            filter_complex.append('[a1][a2]amix=inputs=2:duration=first:dropout_transition=0[aout]')
                        else:
                            filter_complex.append('[a1]volume=1.0[aout]')
                    

                    
                    # 添加滤镜复杂度参数
                    audio_cmd.extend([
                        '-filter_complex',
                        ';'.join(filter_complex)
                    ])
                    
                    # 添加输出参数
                    audio_cmd.extend([
                        '-map', '[aout]',
                        '-f', 'mp4',           # 明确指定输出格式
                        '-c:a', 'aac',
                        '-b:a', '192k',
                        '-ar', '48000',
                        '-ac', '2',
                        '-strict', 'experimental',  # 允许实验性编码器
                        temp_audio
                    ])
                
                    
                    # 执行音频混合
                    self.logger.info(f"Executing audio mixing command: {' '.join(audio_cmd)}")
                    audio_result = subprocess.run(audio_cmd, capture_output=True, text=True)
                    
                    if audio_result.returncode != 0:
                        self.logger.error(f"Audio mixing error: {audio_result.stderr}")
                        raise RuntimeError(f"Audio mixing failed: {audio_result.stderr}")
                    
                    # 最终视频处理命令
                    final_cmd = [
                        'ffmpeg', '-y',
                        '-i', video_path,
                        '-i', temp_audio,
                        '-c:v', 'libx264',
                        '-preset', 'medium',
                        '-crf', '23'
                    ]
                    
                    if filter_chains:
                        final_cmd.extend([
                            '-filter_complex',
                            ','.join(filter_chains)
                        ])
                    
                    final_cmd.extend([
                        '-map', '0:v',      # 使用原始视频的视频流
                        '-map', '1:a',      # 使用混合后的音频
                        '-c:a', 'aac',      # 重新编码音频
                        '-b:a', '192k',
                        '-ar', '48000',
                        '-ac', '2',
                        '-movflags', '+faststart',
                        final_output
                    ])


                    
                    # 执行最终视频处理
                    self.logger.info(f"Executing final video command: {' '.join(final_cmd)}")
                    final_result = subprocess.run(final_cmd, capture_output=True, text=True)
                    
                    if final_result.returncode != 0:
                        self.logger.error(f"Final video processing error: {final_result.stderr}")
                        raise RuntimeError(f"Final video processing failed: {final_result.stderr}")
                    
                    # 清理临时文件
                    if os.path.exists(temp_audio):
                        os.remove(temp_audio)
                    
                else:
                    # 如果没有额外音频，只处理视频效果
                    command = [
                        'ffmpeg', '-y',
                        '-i', video_path
                    ]
                    
                    if filter_chains:
                        command.extend([
                            '-filter_complex',
                            ','.join(filter_chains)
                        ])
                    
                    command.extend([
                        '-c:v', 'libx264',
                        '-preset', 'medium',
                        '-crf', '23',
                        '-c:a', 'aac',
                        '-b:a', '192k',
                        '-ar', '48000',
                        '-ac', '2',
                        '-movflags', '+faststart',
                        final_output
                    ])
                    
                    self.logger.info(f"Executing video command: {' '.join(command)}")
                    result = subprocess.run(command, capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        self.logger.error(f"Video processing error: {result.stderr}")
                        raise RuntimeError(f"Video processing failed: {result.stderr}")
                
                # 验证输出文件
                if not os.path.exists(final_output):
                    raise FileNotFoundError(f"Output file was not created: {final_output}")
                if os.path.getsize(final_output) == 0:
                    raise ValueError(f"Output file is empty: {final_output}")

                self.logger.info("✅ Video processing completed successfully")
                return final_output, video_duration

            except Exception as e:
                self.logger.error(f"❌ Error processing: {str(e)}")
                if isinstance(e, subprocess.CalledProcessError):
                    self.logger.error(f"Command output: {e.output}")
                    self.logger.error(f"Command stderr: {e.stderr}")
                return None, None

        except Exception as e:
            self.logger.error(f"Error applying effects to video: {str(e)}")
            return None, None
            

    def _get_bgm_path(self, bgm_id: str) -> Optional[str]:
        """获取背景音乐文件路径"""
        try:
            # 假设背景音乐文件存储在 assets/bgm 目录下
            bgm_dir = Path('assets/bgm')
            bgm_path = bgm_dir / f"{bgm_id}.mp3"
            
            if bgm_path.exists():
                return str(bgm_path)
            
            self.logger.warning(f"BGM file not found: {bgm_path}")
            return None
        except Exception as e:
            self.logger.error(f"Error getting BGM path: {str(e)}")
            return None

    def _get_encoding_args(self, use_gpu: bool = True) -> dict:
        """Get encoding arguments based on whether GPU encoding is available"""
        if use_gpu:
            return {
                'c:v': 'h264_nvenc',
                'preset': 'p4',
                'tune': 'hq',
                'rc': 'vbr',
                'cq': '23',
                'b:v': '5M',
                'maxrate': '10M',
                'bufsize': '10M',
                'c:a': 'aac',
                'b:a': '192k',
                'ac': 2,
                'ar': '48000',
                'movflags': '+faststart'
            }
        else:
            return {
                'c:v': 'libx264',
                'preset': 'medium',
                'crf': '23',
                'c:a': 'aac',
                'b:a': '192k',
                'ac': 2,
                'ar': '48000',
                'movflags': '+faststart'
            }

        

    def _generate_subtitle_segments(self, narration_content: str, video_duration: float) -> List[dict]:
        """生成字幕段落"""
        sentences = []
        pattern = r'([^。！？.!?]+[。！？.!?])'
        matches = re.finditer(pattern, narration_content)
        
        current_pos = 0
        for match in matches:
            sentence = match.group(1).strip()
            if sentence:
                sentences.append(sentence)
            current_pos = match.end()
        
        if current_pos < len(narration_content):
            last_sentence = narration_content[current_pos:].strip()
            if last_sentence:
                sentences.append(last_sentence)
                
        # 计算每个字的时间
        total_chars = sum(len(s) for s in sentences)
        time_per_char = video_duration / total_chars
        
        # 生成时间段
        segments = []
        current_time = 0
        for sentence in sentences:
            duration = len(sentence) * time_per_char
            end_time = min(current_time + duration, video_duration)
            
            if end_time > current_time:
                segments.append({
                    'text': sentence,
                    'start': current_time,
                    'end': end_time
                })
                current_time = end_time + 0.1
                
        return segments

    def _segments_to_srt(self, segments):
        """Convert subtitle segments to SRT format"""
        srt_content = ""
        for i, segment in enumerate(segments, 1):
            # Convert seconds to SRT time format (HH:MM:SS,mmm)
            start_time = self._seconds_to_srt_time(segment['start'])
            end_time = self._seconds_to_srt_time(segment['end'])
            
            srt_content += f"{i}\n{start_time} --> {end_time}\n{segment['text']}\n\n"
        return srt_content

    def _seconds_to_srt_time(self, seconds):
        """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int((seconds % 1) * 1000)
        seconds = int(seconds)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"




    def _is_valid_transcription(self, transcription_result):
        """检查转写结果是否有效"""
        if not transcription_result or not transcription_result.get('transcription'):
            return False
        
        text = ' '.join([segment['text'] for segment in transcription_result['transcription']])
        return len(text) > 50 and len(transcription_result['transcription']) > 5

    def _generate_audio(self, content, preset, project_no):
        """
        生成音频文件
        
        Args:
            content (str): 需要转换为语音的文本内容
            preset (dict): 预设配置，包含语音设置
            project_no (str): 项目编号
            
        Returns:
            str: 本地音频文件路径
            
        Raises:
            Exception: 当音频生成失败或文件不存在时抛出异常
        """
        try:
            if not content:
                raise ValueError("Content cannot be empty")
                
            # 获取语音映射
            voice_map = self.remote_llm_util.get_voice_id()
            if not voice_map:
                raise Exception("Failed to get voice mapping")
            
            # 选择语音ID
            voice_id = self._select_voice_id(
                voice_map,
                preset.get('narrationVoice', 0),
                preset.get('narrationStyle', 0)
            )


            audio_url = None
            audio_cos_path = None
            # 生成音频
            if voice_id:
                self.logger.info(f'Using ElevenLabs voice ID: {voice_id}')
                response = self.remote_llm_util.text_to_speech_by_elevenlabs(content, voice_id=voice_id)
                audio_url = response.get('audio_url')
                audio_cos_path = response.get('audio_cos_path')
                self.logger.info(f'Audio URL: {audio_url}')
                self.logger.info(f'Audio COS path: {audio_cos_path}')

            # 验证远程路径
            if not audio_url:
                raise Exception("Failed to generate audio: remote path is empty")
                
            # 验证远程文件是否存在
            if not self.cos_util.check_file_exists( audio_cos_path):
                raise Exception(f"Remote audio file does not exist: {audio_cos_path}")

            # 准备本地路径
            audio_filename = os.path.basename(audio_cos_path)
            if not audio_filename:
                raise Exception("Invalid remote path: cannot extract filename")
                
            local_path = os.path.join(os.getcwd(), 'temp', project_no, audio_filename)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            # 下载文件
            try:
                self.cos_util.download_file(
                    audio_cos_path,
                    local_path
                )
            except Exception as e:
                self.logger.error(f"Failed to download audio file: {str(e)}")
                raise Exception(f"Failed to download audio file: {str(e)}")

            # 验证本地文件
            if not os.path.exists(local_path):
                raise Exception(f"Local audio file not found after download: {local_path}")
                
            if os.path.getsize(local_path) == 0:
                raise Exception(f"Downloaded audio file is empty: {local_path}")

            self.logger.info(f"Successfully generated and downloaded audio file: {local_path}")
            return local_path

        except Exception as e:
            error_msg = f"Error generating audio for project {project_no}: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)

    

    def sync_clips_with_audio(self, clips, audio_duration,preset):
        """将视频片段与音频时长同步，使用LLM智能调整"""
        total_video_duration = sum(clip['end'] - clip['start'] for clip in clips)
        
        lang = preset.get('narrationLang', 'ja')
    
        # 多语言提示模板
        prompts = {
            'zh': f'''从原始视频片段中选择合适片段（不可重复选择）：

    当前状态：
    - 音频时长：{audio_duration:.2f}秒
    - 视频总时长：{total_video_duration:.2f}秒

    原始片段详情：
    {json.dumps([{
        'index': i,
        'duration': clip['end'] - clip['start'],
        'start': clip['start'],
        'end': clip['end'],
        'text': clip.get('text', '')
    } for i, clip in enumerate(clips)], ensure_ascii=False, indent=2)}

    核心要求：
    1. 所选片段的总时长必须大于等于音频时长（{audio_duration:.2f}秒），且尽量接近音频时长
       - 总时长绝对不能小于音频时长
       - 总时长应尽可能接近音频时长，避免选取过多导致总时长远大于音频时长

    其他要求：
    2. 每个片段只能选择一次，不能重复使用
    3. 必须完整使用原始片段，不能修改start和end时间
    4. 选择的片段必须均匀分布在整个视频中，不能集中在开始部分
    5. 如果需要选择N个片段，应该大致每隔 总片段数/N 个片段选择一个
    6. 保持内容的连贯性和完整性

    返回格式：
    {{"transcription": [
        {{"start": 必须使用原始start时间, "end": 必须使用原始end时间, "text": "原始文本"}},
        ...
    ]}}''',

            'en': f'''Select appropriate segments from original clips (no duplicates allowed):

    Current status:
    - Audio duration: {audio_duration:.2f} seconds
    - Total video duration: {total_video_duration:.2f} seconds

    Original clip details:
    {json.dumps([{
        'index': i,
        'duration': clip['end'] - clip['start'],
        'start': clip['start'],
        'end': clip['end'],
        'text': clip.get('text', '')
    } for i, clip in enumerate(clips)], ensure_ascii=False, indent=2)}

    Core requirement:
    1. Total duration of selected segments MUST be greater than or equal to audio length ({audio_duration:.2f} seconds), and should be as close as possible to it
       - Total duration MUST NOT be less than audio length
       - Total duration should be close to audio length, avoid selecting too many segments

    Other requirements:
    2. Each segment can only be selected once, no duplicates allowed
    3. Must use complete original segments, no modifications to start and end times
    4. Selected segments must be evenly distributed throughout the video
    5. For N segments to select, choose approximately one segment every (total segments/N)
    6. Maintain content coherence and completeness

    Return format:
    {{"transcription": [
        {{"start": must use original start time, "end": must use original end time, "text": "original text"}},
        ...
    ]}}''',

            'ja': f'''元のクリップから適切なセグメントを選択（重複選択不可）：

    現状：
    - 音声の長さ：{audio_duration:.2f}秒
    - 総ビデオ時間：{total_video_duration:.2f}秒

    元のクリップ詳細：
    {json.dumps([{
        'index': i,
        'duration': clip['end'] - clip['start'],
        'start': clip['start'],
        'end': clip['end'],
        'text': clip.get('text', '')
    } for i, clip in enumerate(clips)], ensure_ascii=False, indent=2)}

    主要な要件：
    1. 選択したセグメントの総時間は必ず音声の長さ（{audio_duration:.2f}秒）以上かつできるだけ近い長さにすること
       - 総時間は絶対に音声の長さより短くしてはいけない
       - 総時間は音声の長さにできるだけ近づける（長すぎないように注意）

    その他の要件：
    2. 各セグメントは一度だけ選択可能、重複選択は不可
    3. 元のセグメントをそのまま使用し、startとend時間の変更は不可
    4. 選択されたセグメントはビデオ全体に均等に分布する必要がある
    5. N個のセグメントを選択する場合、おおよそ（総セグメント数/N）個ごとに1つ選択
    6. 内容の一貫性と完全性を維持

    返却形式：
    {{"transcription": [
        {{"start": 元のstart時間をそのまま使用, "end": 元のend時間をそのまま使用, "text": "元のテキスト"}},
        ...
    ]}}'''
        }

        # 获取对应语言的提示语
        prompt = prompts.get(lang, prompts['ja'])
        # 获取LLM的建议
        response = self.remote_llm_util.native_structured_chat(prompt)
        try:
            self.logger.info(f"response: {response}")
                # 4. 验证和处理结果
            if not response or 'transcription' not in response:
                raise ValueError("LLM返回的结果格式无效")
            
            synced_clips = response.get('transcription', [])
            
            self.logger.info(f"synced_clips: {synced_clips}")
            return synced_clips
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing LLM response: {e}")
            raise

            
    def _fallback_sync_clips(self, clips, audio_duration):
        """降级处理：简单的比例调整"""
        total_video_duration = sum(clip['end'] - clip['start'] for clip in clips)
        adjustment_ratio = audio_duration / total_video_duration
        synced_clips = []
        
        accumulated_time = 0
        for clip in clips:
            new_clip = clip.copy()
            original_duration = clip['end'] - clip['start']
            adjusted_duration = original_duration * adjustment_ratio
            
            new_clip['start'] = round(accumulated_time, 3)
            new_clip['end'] = round(accumulated_time + adjusted_duration, 3)
            
            synced_clips.append(new_clip)
            accumulated_time += adjusted_duration
            
        return synced_clips
    
    '''
    开始用LLM进行分析
    '''
    def select_relevant_transcription(self, transcription, clip_desc, preset):
        """
        Select relevant transcription segments based on clip description and language
        
        Args:
            transcription (dict): Original transcription data
            clip_desc (str): Clip description
            preset (dict): Preset settings including language
            
        Returns:
            dict: Selected transcription segments in the required format
            None: If processing fails
        """
        try:
            lang = preset.get('narrationLang', 'ja')
            
            # 根据语言设置提示语
            prompts = {
                'zh': {
                    'main': '''根据剪辑描述，从以下待剪辑片段中选择符合要求的音频文本。返回选中的片段，保持原始格式不变,按照要求的JSON格式返回。

                    待剪辑片段:
                    {transcription}

                    剪辑描述: {clip_desc}

                    约束：
                    - 返回格式必须与输入的"待剪辑片段"格式完全一致
                    - 只返回你认为符合剪辑要求的片段
                    - 可以选择一个或多个文件中的片段
                    - 每个文件可以选择一个或多个片段
                    - 所有文本必须是中文
                    - 不要添加任何额外解释或修改原始数据结构
                    ''',
                    'error': '输出必须是中文'
                },
                'en': {
                    'main': '''Based on the clip description, select relevant audio text segments from the following clips. Return the selected segments in the original format as required JSON.

                    Clips to process:
                    {transcription}

                    Clip description: {clip_desc}

                    Constraints:
                    - Return format must match the input format exactly
                    - Only return segments that match the requirements
                    - Can select segments from one or multiple files
                    - Can select one or multiple segments per file
                    - All text must be in English
                    - Do not add any extra explanations or modify the original data structure
                    ''',
                    'error': 'Output must be in English'
                },
                'ja': {
                    'main': '''クリップの説明に基づいて、以下のクリップから関連する音声テキストセグメントを選択してください。選択したセグメントを元の形式で必要なJSONとして返してください。

                    処理するクリップ:
                    {transcription}

                    クリップの説明: {clip_desc}

                    制約：
                    - 返却フォーマットは入力フォーマットと完全に一致する必要があります
                    - 要件に一致するセグメントのみを返してください
                    - 1つまたは複数のファイルからセグメントを選択できます
                    - ファイルごとに1つまたは複数のセグメントを選択できます
                    - すべてのテキストは日本語である必要があります
                    - 追加の説明を加えたり、元のデータ構造を変更したりしないでください
                    ''',
                    'error': '出力は日本語である必要があります'
                }
            }
            
            # 获取对应语言的提示语
            prompt_template = prompts.get(lang, prompts['ja'])
            
            # 格式化提示语
            user_prompt = prompt_template['main'].format(
                transcription=json.dumps(transcription, ensure_ascii=False, indent=2),
                clip_desc=clip_desc
            )
            
            try:
                # 调用native OpenAI chat接口
                response = self.remote_llm_util.native_structured_chat(
                    message=user_prompt
                )
                
                if response and hasattr(response, 'parsed'):
                    # 将Pydantic模型转换为字典
                    result = response.parsed.model_dump()
                        
                    self.logger.info(f"====> Selected transcription segments: {result}")
                    return result
                else:
                    self.logger.error("Invalid response format received")
                    return None
                
            except Exception as e:
                self.logger.error(f"Error in selecting transcription segments: {str(e)}")
                return None
        
        except Exception as e:
            self.logger.error(f"Error in transcription selection: {str(e)}")
            return None


    
    def generate_title(self, selected_clips, preset):
        """
        根据视频内容生成标题，支持多语言
        
        Args:
            transcription (dict): 视频文本内容
            preset (dict): 预设配置，包含语言设置
            
        Returns:
            str: 生成的标题
            None: 如果生成失败
        """
        try:
            lang = preset.get('narrationLang', 'ja')

            # 提取所有文本内容
            clip_texts = [clip['text'] for clip in selected_clips if clip.get('text')]
            content = '\n'.join(clip_texts)
        
            
            # 多语言提示模板
            prompts = {
                'zh': {
                    'main': '''根据以下视频内容，生成一个引人入胜、富有吸引力的标题。

                    视频内容:
                    {content}

                    约束：
                    - 标题必须是中文
                    - 标题应简洁有力，不超过20个字
                    - 使用吸引眼球的词语，激发好奇心
                    - 突出内容的独特性和价值
                    - 只使用基本文字，不使用特殊符号
                    - 不使用表情符号和装饰符号
                    - 避免过度夸张的营销用语
                    - 直接返回标题文本
                    ''',
                    'error': '标题必须是中文'
                },
                'en': {
                    'main': '''Create an engaging and captivating title based on the following video content.

                    Video content:
                    {content}

                    Constraints:
                    - Title must be in English
                    - Keep it concise, no more than 10 words
                    - Use attention-grabbing words that spark curiosity
                    - Highlight unique value and appeal
                    - Use only basic text characters
                    - No emojis or special characters
                    - Avoid overly promotional language
                    - Return only the title text
                    ''',
                    'error': 'Title must be in English'
                },
                'ja': {
                    'main': '''以下の動画内容に基づいて、人目を引く魅力的なタイトルを生成してください。

                    動画内容:
                    {content}

                    制約：
                    - タイトルは日本語で書かれている必要があります
                    - 簡潔で力強く、20文字以内にしてください
                    - 興味を引く言葉を使用し、好奇心を刺激すること
                    - コンテンツの独自性と価値を強調すること
                    - 基本的な文字のみを使用すること
                    - 絵文字や装飾記号を使用しないこと
                    - 過度な宣伝的な表現を避けること
                    - タイトルのテキストのみを返すこと
                    ''',
                    'error': 'タイトルは日本語である必要があります'
                }
            }
            
            prompt_template = prompts.get(lang, prompts['ja'])
            
            # 格式化提示语
            user_prompt = prompt_template['main'].format(content=content)
        
            # 调用OpenAI chat接口
            title = self.remote_llm_util.native_chat(user_prompt)
                
            # 清理标题文本（去除引号、空格等）
            title = self._clean_title(title, lang)
            
            self.logger.info(f"Generated title: {title}")
            return title
            
        except Exception as e:
            self.logger.error(f"Error generating title: {str(e)}")
            return None
            

        
    def generate_description(self, selected_clips, preset):
        """
        根据视频内容生成描述，支持多语言
        
        Args:
            transcription (dict): 视频文本内容
            preset (dict): 预设配置，包含语言设置
            
        Returns:
            str: 生成的描述
            None: 如果生成失败
        """
        try:
            lang = preset.get('narrationLang', 'ja')

            # 提取所有文本内容
            clip_texts = [clip['text'] for clip in selected_clips if clip.get('text')]
            content = '\n'.join(clip_texts)
        
            
            # 多语言提示模板
            prompts = {
                'zh': {
                    'main': '''以小红书博主的风格，基于以下内容写一篇吸引人的软文。以生动有趣的语言描述，突出独特魅力和情感共鸣。行文自然流畅，至少包含三个重点内容。适当使用表情增加趣味性。

                    模型识别的画面内容：{content}
                    
                    注意：
                    - 使用基本文字和常用表情符号
                    - 每段落可以使用1-2个表情，不要过度使用
                    - 只使用UTF-8MB4支持的表情符号（如：😊 🌟 ✨ 💖 👉 等）
                    - 保持文字简洁清晰
                    - 不要输出标题，正文等开始的标志
                    - 直接输出完整的软文内容
                    ''',
                    'error': '描述必须是中文'
                },
                'en': {
                    'main': '''Write an engaging social media post in the style of a lifestyle blogger based on the following content. Use vivid and interesting language to create emotional resonance. The writing should flow naturally and cover at least three key points. Add appropriate emojis to enhance engagement.

                    Model detected scene content:
                    {content}
                    
                    Note:
                    - Use basic text and common emojis
                    - Use 1-2 emojis per paragraph, avoid overuse
                    - Only use UTF-8MB4 supported emojis (e.g., 😊 🌟 ✨ 💖 👉)
                    - Keep the text clean and simple
                    - Do not output format markers like "title" or "body"
                    - Output the complete article directly
                    ''',
                    'error': 'Description must be in English'
                },
                'ja': {
                    'main': '''ライフスタイルブロガーのスタイルで、以下の内容に基づいて魅力的な記事を作成してください。生き生きとした興味深い言葉で感動を呼び起こし、少なくとも3つの重要なポイントを含む自然な文章で表現してください。適切な絵文字を使用して魅力を高めてください。

                    モデルが認識した画面内容：
                    {content}
                    
                    注意：
                    - 基本的な文字と一般的な絵文字を使用
                    - 段落ごとに1-2個の絵文字を使用し、過度な使用は避ける
                    - UTF-8MB4対応の絵文字のみを使用（例：😊 🌟 ✨ 💖 👉）
                    - テキストはシンプルに保つ
                    - 「タイトル」や「本文」などの形式マーカーを出力しない
                    - 記事を直接出力する
                    ''',
                    'error': '説明文は日本語である必要があります'
                }
            }
            
            # 获取对应语言的提示语
            prompt_template = prompts.get(lang, prompts['ja'])
            
            # 格式化提示语
            user_prompt = prompt_template['main'].format(content=content)
            
            # 调用OpenAI chat接口
            description = self.remote_llm_util.native_chat(user_prompt)
            
            # 清理和格式化描述文本
            if os.environ.get('IS_FORMAT_DESCRIPTION', 'false') == 'true':
                description = self._format_description(description, lang)
            
            # 验证格式化后的文本是否可以正确序列化
            try:
                json.dumps({"description": description}, ensure_ascii=False)
                self.logger.info(f"Generated description: {description}")
                return description
            except (TypeError, ValueError) as e:
                self.logger.error(f"Description serialization failed: {str(e)}")
                return None
            
        except Exception as e:
            self.logger.error(f"Error generating description: {str(e)}")
            return None
            
    

    def _format_description(self, description, lang):
        """
        格式化描述文本，确保输出格式一致性
        
        Args:
            description (str): 原始描述文本
            lang (str): 语言代码
            
        Returns:
            str: 格式化后的描述文本
        """
        if not description:
            return ""
            
        try:
            # 基础清理
            description = description.strip()
            description = re.sub(r'[\u200b\ufeff\u200d]', '', description)  # 移除零宽字符
            
            # 统一换行符
            description = description.replace('\r\n', '\n').replace('\r', '\n')
            
            # 移除多余的引号
            description = description.strip('"').strip('"').strip('"')
            
            # 根据语言进行特定处理
            if lang == 'ja':
                # 处理日文句号和换行
                sentences = []
                current = []
                
                for char in description:
                    current.append(char)
                    if char == '。':
                        sentences.append(''.join(current))
                        current = []
                
                if current:  # 处理最后一个没有句号的部分
                    sentences.append(''.join(current))
                
                # 过滤空句子并合并
                description = '\n'.join(s.strip() for s in sentences if s.strip())
                
            elif lang == 'zh':
                # 处理中文句号
                sentences = re.split(r'[。！？]', description)
                sentences = [s.strip() for s in sentences if s.strip()]
                description = '。\n'.join(sentences) + '。'
                
            elif lang == 'en':
                # 处理英文句号
                sentences = re.split(r'(?<=[.!?])\s+', description)
                sentences = [s.strip() for s in sentences if s.strip()]
                description = '\n'.join(sentences)
            
            # 最终清理
            description = re.sub(r'\n+', '\n', description)  # 移除多余的换行
            description = description.strip()  # 移除首尾空白
            
            return description
            
        except Exception as e:
            self.logger.error(f"Error formatting description: {str(e)}")
            return description.strip()  # 出错时返回简单处理的文本


    def generate_narration_directly(self,selected_clips: list, preset: dict) -> Optional[str]:
        """根据视频内容生成TikTok风格的旁白"""
        try:
            lang = preset.get('narrationLang', 'ja')
            narration_style = preset.get('narrationStyle', 0)
            # 获取风格描述
            style_translations = {
                'zh': {0: '', 1: '慈祥温和', 2: '文静优雅', 3: '轻松幽默', 4: '雄壮有力', 5: '和蔼可亲'},
                'en': {0: '', 1: 'warm and gentle', 2: 'calm and elegant', 3: 'light and humorous', 
                    4: 'strong and powerful', 5: 'kind and friendly'},
                'ja': {0: '', 1: '温かく優しい', 2: '落ち着いた上品', 3: '軽やかでユーモア', 
                    4: '力強い', 5: '親しみやすい'}
            }
            style_tone = style_translations.get(lang, {}).get(narration_style, '')

            # 生成旁白提示
            narration_prompts = {
                'zh': f'''基于以下视频片段生成连贯的旁白：

    视频片段：
    {json.dumps([{
        'time': f"{clip['end']}-{clip['start']}秒",
        'content': clip['text']
    } for clip in selected_clips], ensure_ascii=False, indent=2)}

    要求：
    1. 使用{style_tone}语气
    2. 分两步处理：
       第一步：为每个片段单独生成对应的旁白句子
       第二步：将所有句子组合并优化，确保整体逻辑连贯流畅
    3. 语言要求：
       - 每个片段的旁白要与内容紧密相关
       - 使用过渡词连接不同片段
       - 避免重复词语
       - 保持语气统一
    4. 时长控制：
       - 确保每个旁白句子的长度与对应视频片段时长相匹配
       - 控制语速自然，不要过快或过慢
    5. 格式要求：
       - 只输出纯文本
       - 每句话用句号结尾
       - 不使用其他标点符号
       - 不添加任何特殊字符或表情
    6. 结构要求：
       - 开头要吸引注意力
       - 中间部分要层层递进
       - 结尾要有总结或升华
       - 整体符合起承转合结构

    输出格式：
    完整的旁白文本，句子之间用句号分隔。''',

                'en': f'''Generate coherent narration based on the following video segments:

    Video segments:
    {json.dumps([{
        'time': f"{clip['end']}-{clip['start']} seconds",
        'content': clip['text']
    } for clip in selected_clips], ensure_ascii=False, indent=2)}

    Requirements:
    1. Use {style_tone} tone
    2. Two-step process:
       First: Generate individual narration for each segment
       Second: Combine and optimize all sentences for overall coherence
    3. Language requirements:
       - Each segment's narration must closely relate to content
       - Use transition words between segments
       - Avoid word repetition
       - Maintain consistent tone
    4. Duration control:
       - Match each narration length to video segment duration
       - Maintain natural speaking pace
    5. Format requirements:
       - Output plain text only
       - End each sentence with a period
       - No other punctuation marks
       - No special characters or emoji
    6. Structure requirements:
       - Engaging opening
       - Progressive development
       - Proper transitions
       - Meaningful conclusion

    Output format:
    Complete narration text with sentences separated by periods.''',

                'ja': f'''以下の動画セグメントに基づいて一貫性のあるナレーションを生成：

    動画セグメント：
    {json.dumps([{
        'time': f"{clip['end']}-{clip['start']}秒",
        'content': clip['text']
    } for clip in selected_clips], ensure_ascii=False, indent=2)}

    要件：
    1. {style_tone}口調を使用
    2. 二段階処理：
       第一段階：各セグメントに対する個別のナレーションを生成
       第二段階：全文章を組み合わせて最適化し、全体の一貫性を確保
    3. 言語要件：
       - 各セグメントのナレーションは内容に密接に関連
       - セグメント間に接続語を使用
       - 単語の重複を避ける
       - 一貫した口調を維持
    4. 時間制御：
       - 各ナレーションの長さを動画セグメントの長さに合わせる
       - 自然な話速を維持
    5. フォーマット要件：
       - プレーンテキストのみ出力
       - 各文を句点で終える
       - 他の句読点は使用しない
       - 特殊文字や絵文字は使用しない
    6. 構造要件：
       - 注目を集める導入
       - 段階的な展開
       - 適切な転換
       - 意味のある結論

    出力形式：
    句点で区切られた完全なナレーションテキスト。'''
            }

            self.logger.info(f"lang is {lang}")
            # 生成旁白
            narration = self.remote_llm_util.native_chat(narration_prompts.get(lang, narration_prompts['ja']))
            self.logger.info(f"生成的旁白NARRATION: {narration}")
            
            # 清理并返回旁白文本
            if os.environ.get('IS_CLEAN_NARRATION', 'false') == 'true':
                narration = self._clean_narration_text(narration, lang)
                self.logger.info(f"生成的cleared旁白NARRATION: {narration}")
   
            return narration
            
        except ffmpeg.Error as e:
            self.logger.error(f"FFmpeg error: {e.stderr.decode('utf8') if e.stderr else str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Error generating narration: {str(e)}")
            return None
        


    def generate_narration(self, raw_merged_video_path: str, selected_clips: list, preset: dict) -> Optional[str]:
        """根据视频内容生成TikTok风格的旁白"""
        try:
            lang = preset.get('narrationLang', 'ja')
            narration_style = preset.get('narrationStyle', 0)
            
            # 使用 ffprobe 获取视频信息
            probe = ffmpeg.probe(raw_merged_video_path)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            video_duration = float(probe['format']['duration'])
            
            self.logger.info(f"视频时长: {video_duration:.2f} seconds")

            # 计算目标字符数
            chars_per_second = {'zh': 3, 'en': 4, 'ja': 4}.get(lang, 3)
            target_length = int(video_duration * chars_per_second)
            
            # 获取风格描述
            style_translations = {
                'zh': {0: '', 1: '慈祥温和', 2: '文静优雅', 3: '轻松幽默', 4: '雄壮有力', 5: '和蔼可亲'},
                'en': {0: '', 1: 'warm and gentle', 2: 'calm and elegant', 3: 'light and humorous', 
                    4: 'strong and powerful', 5: 'kind and friendly'},
                'ja': {0: '', 1: '温かく優しい', 2: '落ち着いた上品', 3: '軽やかでユーモア', 
                    4: '力強い', 5: '親しみやすい'}
            }
            style_tone = style_translations.get(lang, {}).get(narration_style, '')

            self.logger.info(f'====>剪辑出来的片段: {selected_clips}')
            
            # 生成旁白提示
            narration_prompts = {
                'zh': f'''基于以下视频片段生成连贯的旁白，每个片段对应一句话：

    视频片段：
    {json.dumps([{
        'time': f"{clip['end']}-{clip['start']}秒",
        'content': clip['text']
    } for clip in selected_clips], ensure_ascii=False, indent=2)}

    规则：
    1. 使用{style_tone}语气
    2. 总字数严格控制在{target_length}字
    3. 为每个时间片段生成对应的旁白句子
    4. 每句话用句号结尾
    5. 所有句子之间要逻辑连贯
    6. 按照起承转合的结构组织整体内容
    7. 只输出纯文本内容
    8. 不要加任何额外标点
    9. 不要加任何特殊字符和表情符号
    10. 不要输出任何额外说明
    11. 确保每句话的长度与对应时间片段的长度相匹配

    格式要求：
    第一个片段的旁白。第二个片段的旁白。第三个片段的旁白。''',

                'en': f'''Generate coherent narration for each video segment:

    Video segments:
    {json.dumps([{
        'time': f"{clip['end']}-{clip['start']} seconds",
        'content': clip['text']
    } for clip in selected_clips], ensure_ascii=False, indent=2)}

    Rules:
    1. Use {style_tone} tone
    2. Total length strictly {target_length} characters
    3. Generate corresponding narration for each time segment
    4. End each sentence with a period
    5. Maintain logical flow between all sentences
    6. Follow introduction-development-transition-conclusion structure
    7. Output plain text only
    8. No additional punctuation
    9. No special characters or emoji
    10. No additional explanations
    11. Ensure each sentence length matches its time segment duration

    Format:
    First segment narration. Second segment narration. Third segment narration.''',

                'ja': f'''各ビデオセグメントに対応する一貫性のあるナレーションを生成：

    ビデオセグメント：
    {json.dumps([{
        'time': f"{clip['end']}-{clip['start']}秒",
        'content': clip['text']
    } for clip in selected_clips], ensure_ascii=False, indent=2)}

    ルール：
    1. {style_tone}口調を使用
    2. 合計文字数を厳密に{target_length}文字に
    3. 各時間セグメントに対応するナレーションを生成
    4. 各文を句点で終える
    5. 文章間の論理的なつながりを保つ
    6. 起承転結の構造に従う
    7. プレーンテキストのみを出力
    8. 追加の句読点なし
    9. 特殊文字や絵文字なし
    10. 追加の説明なし
    11. 各文の長さを対応する時間セグメントの長さに合わせる

    フォーマット：
    最初のセグメントのナレーション。2番目のセグメントのナレーション。3番目のセグメントのナレーション。'''
            }

            # 生成旁白
            narration = self.remote_llm_util.native_chat(narration_prompts.get(lang, narration_prompts['ja']))
            self.logger.info(f"=====> 生成的旁白NARRATION: {narration}")
            
            # 清理并返回旁白文本
            narration = self._clean_narration_text(narration, lang)
            self.logger.info(f"生成的cleared旁白NARRATION: {narration}")
            self.logger.info(f"生成的旁白 ({len(narration)}字): {narration}")
            
            return narration
            
        except ffmpeg.Error as e:
            self.logger.error(f"FFmpeg error: {e.stderr.decode('utf8') if e.stderr else str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Error generating narration: {str(e)}")
            return None

    def _clean_narration_text(self, text: str, lang: str) -> str:
        """清理旁白文本"""
        if not text:
            return ""
            
        # 基本清理
        text = text.strip()
        
        # 语言特定的清理规则
        if lang == 'zh':
            # 移除中文标点符号，保留句号
            text = re.sub(r'[！？，、；：""''【】（）]', '', text)
        elif lang == 'en':
            # 移除英文标点符号，保留句号
            text = re.sub(r'[!?,;:"\'()\[\]]', '', text)
        elif lang == 'ja':
            # 移除日文标点符号，保留句号
            text = re.sub(r'[！？、；：「」『』（）]', '', text)
        
        # 统一句号并移除多余空格
        text = re.sub(r'[.。]+', '。', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def _clean_narration_text(self, text, lang):
        """
        清理旁白文本中的特殊字符，保留主要内容
        
        Args:
            text (str): 原始旁白文本
            lang (str): 语言代码
                
        Returns:
            str: 清理后的文本
        """
        if not text or len(text.strip()) == 0:
            self.logger.warning("\033[93mEmpty narration text received\033[0m")
            return ""
                
        try:
            # 基础清理
            text = text.strip()
            
            # 只移除特殊字符，保留引号内的内容
            symbols_to_remove = [
                '{', '}', '[', ']', '(', ')', '【', '】', '（', '）',
                '*', '#', '@', '$', '%', '^', '&', '+', '=', '|', '\\',
                '~', '<', '>', 
            ]
            for symbol in symbols_to_remove:
                text = text.replace(symbol, '')
                        
            # 语言特定的处理
            if lang == 'ja':
                # 保留日文常用标点
                allowed_punctuation = '。、！？…：「」『』'
                # 将多个连续的标点替换为单个
                for punct in allowed_punctuation:
                    text = re.sub(f'{punct}+', punct, text)
                        
            elif lang == 'zh':
                # 保留中文常用标点
                allowed_punctuation = '。，！？…：；'
                # 将多个连续的标点替换为单个
                for punct in allowed_punctuation:
                    text = re.sub(f'{punct}+', punct, text)
                        
            elif lang == 'en':
                # 保留英文常用标点
                allowed_punctuation = '.,:!?'
                # 将多个连续的标点替换为单个
                for punct in allowed_punctuation:
                    text = re.sub(f'\\{punct}+', punct, text)
                
            # 移除多余的空格
            text = re.sub(r'\s+', ' ', text)
            # 移除行首行尾的空格
            text = text.strip()
            
            # 移除零宽字符
            text = re.sub(r'[\u200b\ufeff\u200d]', '', text)
            
            return text
                
        except Exception as e:
            self.logger.error(f"\033[93mError cleaning narration text: {str(e)}\033[0m")
            return text.strip() if text else ""
    
    
    def merge_video_clips(self, clip_paths: List[str], merged_video_path: str) -> Optional[str]:
        """
        使用 ffmpeg-python 合并视频片段，支持 GPU 加速
        
        Args:
            clip_paths: 视频片段路径列表
            merged_video_path: 合并后的视频保存路径
            
        Returns:
            str: 合并后的视频路径，失败则返回None
        """
        if not clip_paths:
            raise ValueError("No clip paths provided")

        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(merged_video_path), exist_ok=True)

            # 创建临时文件列表
            temp_list_path = None
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                temp_list_path = f.name
                # 写入文件列表，使用 ffmpeg 的 concat demuxer
                for clip_path in clip_paths:
                    if os.path.exists(clip_path):
                        # 使用绝对路径并转义特殊字符
                        abs_path = os.path.abspath(clip_path)
                        f.write(f"file '{abs_path}'\n")
                    else:
                        self.logger.warning(f"Warning: Clip file not found: {clip_path}")

            if os.path.getsize(temp_list_path) == 0:
                raise ValueError("No valid video clips to merge")

            try:
                # 构建 ffmpeg 命令
                stream = (
                    ffmpeg
                    .input(temp_list_path, f='concat', safe=0)
                    .output(
                        merged_video_path,
                        c='copy',  # 直接复制流，避免重新编码
                        movflags='faststart',  # 支持快速启动播放
                        loglevel='info'
                    )
                )
                
                # 如果需要重新编码（比如视频编码不一致时），使用 GPU 加速：
                # .output(
                #     merged_video_path,
                #     **{
                #         'c:v': 'h264_nvenc',  # NVIDIA GPU 加速
                #         'preset': 'p4',        # NVIDIA 预设
                #         'tune': 'hq',          # 高质量调优
                #         'b:v': '2000k',        # 视频比特率
                #         'maxrate': '4000k',    # 最大比特率
                #         'bufsize': '8000k',    # 缓冲大小
                #         'c:a': 'aac',          # 音频编码器
                #         'b:a': '192k',         # 音频比特率
                #         'ar': 48000,           # 音频采样率
                #         'movflags': 'faststart'
                #     }
                # )

                # 执行合并
                self.logger.info(f"Merging {len(clip_paths)} video clips...")
                stream = ffmpeg.overwrite_output(stream)
                stream.run(capture_stdout=True, capture_stderr=True)

                if os.path.exists(merged_video_path):
                    self.logger.info(f"Video successfully merged and saved to: {merged_video_path}")
                    return merged_video_path
                else:
                    raise RuntimeError("Merged video file not found after processing")

            except ffmpeg.Error as e:
                self.logger.error(f"FFmpeg error: {e.stderr.decode('utf8') if e.stderr else str(e)}")
                return None

        except Exception as e:
            self.logger.error(f"Error during video merging: {str(e)}")
            return None

        finally:
            # 清理临时文件
            if temp_list_path and os.path.exists(temp_list_path):
                try:
                    os.remove(temp_list_path)
                except Exception as e:
                    self.logger.error(f"Warning: Failed to remove temporary file: {e}")

    def check_video_compatibility(self, clip_paths: List[str]) -> bool:
        """
        检查视频片段的编码和格式是否兼容
        """
        if not clip_paths:
            return False

        try:
            codecs = set()
            for clip_path in clip_paths:
                probe = ffmpeg.probe(clip_path)
                video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
                codecs.add(video_info['codec_name'])
            
            # 如果所有视频使用相同的编码器，返回 True
            return len(codecs) == 1

        except Exception as e:
            self.logger.error(f"Error checking video compatibility: {e}")
            return False

    def merge_video_clips_with_transcoding(self, clip_paths: List[str], merged_video_path: str) -> Optional[str]:
        """
        合并视频片段，如果需要则进行转码（使用 GPU 加速）
        """
        if not self.check_video_compatibility(clip_paths):
            try:
                # 创建临时目录
                temp_dir = tempfile.mkdtemp()
                transcoded_paths = []

                # 转码所有视频片段
                for i, clip_path in enumerate(clip_paths):
                    output_path = os.path.join(temp_dir, f"temp_{i}.mp4")
                    try:
                        stream = (
                            ffmpeg
                            .input(clip_path)
                            .output(
                                output_path,
                                **{
                                    'c:v': 'h264_nvenc',  # NVIDIA GPU 加速
                                    'preset': 'p4',
                                    'tune': 'hq',
                                    'b:v': '2000k',
                                    'c:a': 'aac',
                                    'b:a': '192k',
                                }
                            )
                        )
                        stream = ffmpeg.overwrite_output(stream)
                        stream.run(capture_stdout=True, capture_stderr=True)
                        transcoded_paths.append(output_path)
                    except ffmpeg.Error as e:
                        self.logger.error(f"Error transcoding clip {i}: {e.stderr.decode('utf8')}")
                        continue

                # 合并转码后的视频
                if transcoded_paths:
                    return self.merge_video_clips(transcoded_paths, merged_video_path)
                else:
                    raise RuntimeError("No clips were successfully transcoded")

            finally:
                # 清理临时文件
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    self.logger.error(f"Warning: Failed to remove temporary directory: {e}")

        else:
            # 如果视频兼容，直接合并
            return self.merge_video_clips(clip_paths, merged_video_path)



    def concatenate_videos(self, clip_mapping: Dict[str, str], clips: List[Dict[str, Any]], 
                         merged_video_path: str) -> Optional[str]:
        """
        使用 ffmpeg-python 连接视频并添加转场效果，支持 GPU 加速
        
        Args:
            clip_mapping: 视频URL到本地文件路径的映射
            clips: 视频片段列表，包含类型和转场效果信息
            merged_video_path: 合并后的视频保存路径
        """
        if not clip_mapping:
            raise ValueError("No clips were successfully downloaded")

        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(merged_video_path), exist_ok=True)
            
            # 创建临时目录
            temp_dir = tempfile.mkdtemp()
            processed_clips = []
            
            for index, clip in enumerate(clips):
                if clip['type'] == 'video':
                    video_url = clip.get('videoUrl')
                    if not video_url or video_url not in clip_mapping:
                        self.logger.error(f"Skip invalid clip {index}: missing URL or mapping")
                        continue
                        
                    video_path = clip_mapping[video_url]
                    if not os.path.exists(video_path):
                        self.logger.error(f"Skip clip {index}: file not found at {video_path}")
                        continue
                    
                    # 处理转场效果
                    transition_effect = clip.get('transitionEffect', 0)
                    temp_output = os.path.join(temp_dir, f"processed_{index}.mp4")
                    
                    try:
                        # 构建基本的视频流
                        stream = ffmpeg.input(video_path)
                        
                        # 应用转场效果
                        if transition_effect in (1, 2):  # 淡入或淡出
                            # 获取视频时长
                            probe = ffmpeg.probe(video_path)
                            duration = float(probe['format']['duration'])
                            
                            # 创建淡入淡出滤镜
                            fade_duration = 0.5
                            if transition_effect == 1:  # 淡入
                                filter_params = f"fade=t=in:st=0:d={fade_duration}"
                            else:  # 淡出
                                fade_start = duration - fade_duration
                                filter_params = f"fade=t=out:st={fade_start}:d={fade_duration}"
                            
                            stream = stream.filter('fps', fps=24, round='up')
                            stream = stream.filter('scale', 'iw', 'ih')
                            stream = stream.filter('format', 'yuv420p')
                            stream = stream.filter('fade', **dict(x.split('=') for x in filter_params.split(':')))
                        
                        # 使用 GPU 加速编码
                        stream = (
                            ffmpeg
                            .output(
                                stream,
                                temp_output,
                                **{
                                    'c:v': 'h264_nvenc',  # NVIDIA GPU 加速
                                    'preset': 'p4',        # NVIDIA 预设
                                    'tune': 'hq',          # 高质量调优
                                    'b:v': '2000k',        # 视频比特率
                                    'maxrate': '4000k',    # 最大比特率
                                    'bufsize': '8000k',    # 缓冲大小
                                    'c:a': 'aac',          # 音频编码器
                                    'b:a': '192k',         # 音频比特率
                                    'ar': 48000,           # 音频采样率
                                }
                            )
                        )
                        
                        # 执行处理
                        stream = ffmpeg.overwrite_output(stream)
                        stream.run(capture_stdout=True, capture_stderr=True)
                        
                        processed_clips.append(temp_output)
                        
                    except ffmpeg.Error as e:
                        self.logger.error(f"Error processing clip {index}: {e.stderr.decode('utf8')}")
                        continue
                        
                elif clip['type'] == 'effect':
                    # 效果片段的处理已经在前一个视频片段中完成
                    continue
                    
            if not processed_clips:
                raise ValueError("No valid video clips to concatenate")
                
            # 创建合并列表文件
            concat_list_path = os.path.join(temp_dir, 'concat_list.txt')
            with open(concat_list_path, 'w') as f:
                for clip_path in processed_clips:
                    f.write(f"file '{clip_path}'\n")
            
            # 最终合并
            try:
                stream = (
                    ffmpeg
                    .input(concat_list_path, f='concat', safe=0)
                    .output(
                        merged_video_path,
                        c='copy',  # 直接复制流以避免重新编码
                        movflags='faststart'
                    )
                )
                
                stream = ffmpeg.overwrite_output(stream)
                stream.run(capture_stdout=True, capture_stderr=True)
                
                self.logger.info(f"Video successfully concatenated and saved to: {merged_video_path}")
                return merged_video_path
                
            except ffmpeg.Error as e:
                self.logger.error(f"Error during final concatenation: {e.stderr.decode('utf8')}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error during video concatenation: {str(e)}")
            return None
            
        finally:
            # 清理临时文件
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                self.logger.error(f"Warning: Failed to remove temporary directory: {e}")

    def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        """获取视频信息"""
        try:
            probe = ffmpeg.probe(video_path)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            return {
                'duration': float(probe['format']['duration']),
                'width': int(video_info['width']),
                'height': int(video_info['height']),
                'fps': eval(video_info['r_frame_rate'])
            }
        except Exception as e:
            self.logger.error(f"Error getting video info: {e}")
            return {}
        
    
    def check_ffmpeg(self):
        try:
            subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except:
            return False

    def ensure_dir(self,file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)



    def generate_video_clips(self, project_no: str, clip_transcription: Dict[str, Any], 
                           download_video_path: str) -> List[str]:
        """
        使用 ffmpeg-python 生成视频片段，支持 GPU 加速
        
        Args:
            project_no: 项目编号
            clip_transcription: 包含时间戳的转录信息
            download_video_path: 源视频路径
            
        Returns:
            List[str]: 生成的视频片段路径列表
        """
        try:
            clip_paths = []
            
            # 处理输入的转录信息
            if isinstance(clip_transcription, str):
                try:
                    # 尝试不同的编码方式解析JSON
                    for encoding in ['utf-8', 'utf-8-sig', 'gbk', 'shift-jis', 'euc-jp']:
                        try:
                            decoded_str = clip_transcription.encode(encoding).decode(encoding)
                            clip_transcription = json.loads(decoded_str)
                            self.logger.info(f"Successfully parsed JSON with {encoding} encoding")
                            break
                        except (UnicodeError, json.JSONDecodeError):
                            continue
                    else:
                        raise ValueError("Failed to parse JSON with any known encoding")
                except Exception as e:
                    self.logger.error(f"Failed to parse clip_transcription: {str(e)}")
                    self.logger.error(f"Original clip_transcription: {repr(clip_transcription)}")
                    raise ValueError(f"Failed to parse clip_transcription: {str(e)}")

            # 验证输入数据
            if not isinstance(clip_transcription, dict):
                raise ValueError(f"clip_transcription must be a dictionary, but got {type(clip_transcription)}")
            
            transcription = clip_transcription.get('transcription', [])
            if not transcription:
                self.logger.error("No transcriptions found in clip_transcription")
                return clip_paths
            
            self.logger.debug(f"Debug: Processing video_path = {download_video_path}")
            
            # 创建输出目录
            video_name = Path(download_video_path).stem
            clips_dir = Path('temp') / str(project_no) / 'clips'
            clips_dir.mkdir(parents=True, exist_ok=True)
            
            # 获取视频信息
            try:
                probe = ffmpeg.probe(download_video_path)
                video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
                total_duration = float(probe['format']['duration'])
            except ffmpeg.Error as e:
                self.logger.error(f"Error probing video: {e.stderr.decode('utf8')}")
                raise
            
            self.logger.debug(f"Debug: timestamps for {video_name} = {json.dumps(transcription, indent=2)}")
            
            # 处理每个时间段
            for i, timestamp in enumerate(transcription):
                if 'start' not in timestamp or 'end' not in timestamp:
                    self.logger.error(f"Warning: Skipping invalid timestamp: {timestamp}")
                    continue
                
                start_time = timestamp['start']
                end_time = timestamp['end']
                
                # 验证时间戳
                if not (0 <= start_time < end_time <= total_duration):
                    self.logger.error(f"Warning: Invalid time range: {start_time}-{end_time}-{total_duration}")
                    continue
                
                try:
                    # 生成输出文件路径
                    clip_filename = f"{video_name}_clip_{i}_{uuid.uuid4()}.mp4"
                    clip_path = str(clips_dir / clip_filename)
                    
                    # 构建 ffmpeg 命令
                    stream = (
                        ffmpeg
                        .input(download_video_path, ss=start_time, t=end_time-start_time)
                        .output(
                            clip_path,
                            **{
                                'c:v': 'h264_nvenc',  # NVIDIA GPU 加速
                                'preset': 'p4',        # NVIDIA 预设
                                'tune': 'hq',          # 高质量调优
                                'b:v': '2000k',        # 视频比特率
                                'maxrate': '4000k',    # 最大比特率
                                'bufsize': '8000k',    # 缓冲大小
                                'c:a': 'aac',          # 音频编码器
                                'b:a': '192k',         # 音频比特率
                                'ar': 48000,           # 音频采样率
                                'vsync': 0,            # 视频同步模式
                                'copyts': None,        # 保持原始时间戳
                                'avoid_negative_ts': 'make_zero',  # 处理负时间戳
                                'async': 1             # 音频同步模式
                            }
                        )
                    )
                    
                    # 执行命令
                    stream = ffmpeg.overwrite_output(stream)
                    stream.run(capture_stdout=True, capture_stderr=True)
                    
                    # 验证生成的片段
                    if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
                        clip_paths.append(clip_path)
                        self.logger.info(f"Saved individual clip: {clip_path}")
                    else:
                        self.logger.error(f"Warning: Failed to generate clip for {start_time}-{end_time}")
                        
                except ffmpeg.Error as e:
                    self.logger.error(f"Error creating clip {i}: {e.stderr.decode('utf8')}")
                    continue
                except Exception as e:
                    self.logger.error(f"Error creating clip {i}: {str(e)}")
                    continue
            
            if not clip_paths:
                raise ValueError("No valid video clips were generated")
            
            self.logger.debug(f"Debug: Number of clips generated: {len(clip_paths)}")
            return clip_paths
            
        except Exception as e:
            self.logger.error(f"Error generating video clips: {str(e)}")
            raise

    def _check_clip_validity(self, clip_path: str) -> bool:
        """验证生成的视频片段是否有效"""
        try:
            probe = ffmpeg.probe(clip_path)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            duration = float(probe['format']['duration'])
            return duration > 0 and int(video_info.get('nb_frames', 0)) > 0
        except Exception:
            return False

    def _get_optimal_encoding_params(self) -> Dict[str, Any]:
        """获取最优编码参数"""
        try:
            # 检查是否支持 NVIDIA GPU 编码
            subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], 
                         capture_output=True, text=True).stdout
            return {
                'c:v': 'h264_nvenc',  # NVIDIA GPU 加速
                'preset': 'p4',
                'tune': 'hq',
                'b:v': '2000k',
                'maxrate': '4000k',
                'bufsize': '8000k',
            }
        except Exception:
            # 回退到 CPU 编码
            return {
                'c:v': 'libx264',
                'preset': 'medium',
                'crf': '23',
                'b:v': '2000k',
            }
    


    '''
    生成切片视频
    '''
    def generate_video_clips_by_local(self, project_no: str, segments: Union[str, dict, list], 
                                    download_video_path: str) -> List[str]:
        """
        使用 ffmpeg-python 根据时间段生成视频片段，支持 GPU 加速
        
        Args:
            project_no: 项目编号
            segments: 时间段信息（JSON字符串、字典或列表）
            download_video_path: 源视频路径
            
        Returns:
            List[str]: 生成的视频片段路径列表
        """
        try:
            clip_paths = []
            self.logger.debug(f"Segments type = {type(segments)}")
            
            # 处理输入的segments
            segments = self._parse_segments(segments)
            
            if not segments:
                self.logger.warning("No valid segments found")
                return clip_paths
            
            self.logger.debug(f"Processing video_path = {download_video_path}")
            
            # 设置输出目录
            video_name = Path(download_video_path).stem
            clips_dir = Path('temp') / str(project_no) / 'clips'
            clips_dir.mkdir(parents=True, exist_ok=True)
            
            # 获取视频信息
            try:
                probe = ffmpeg.probe(download_video_path)
                video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
                total_duration = float(probe['format']['duration'])
                
                # 获取最优编码参数
                encode_params = self._get_optimal_encoding_params()
                
                self.logger.debug(f"Timestamps for {video_name} = {json.dumps(segments, indent=2)}")
                
                # 处理每个时间段
                for i, segment in enumerate(segments):
                    if not self._validate_segment(segment, total_duration):
                        continue
                    
                    start_time = segment['start']
                    end_time = segment['end']
                    
                    try:
                        # 生成输出文件路径
                        clip_filename = f"{video_name}_clip_{i}_{uuid.uuid4()}.mp4"
                        clip_path = str(clips_dir / clip_filename)
                        
                        # 构建 ffmpeg 命令
                        stream = (
                            ffmpeg
                            .input(download_video_path, ss=start_time, t=end_time-start_time)
                            .output(
                                clip_path,
                                **{
                                    **encode_params,
                                    'c:a': 'aac',          # 音频编码器
                                    'b:a': '192k',         # 音频比特率
                                    'ar': 48000,           # 音频采样率
                                    'vsync': 0,            # 视频同步模式
                                    'copyts': None,        # 保持原始时间戳
                                    'avoid_negative_ts': 'make_zero',  # 处理负时间戳
                                    'async': 1             # 音频同步模式
                                }
                            )
                        )
                        
                        # 执行命令
                        stream = ffmpeg.overwrite_output(stream)
                        stream.run(capture_stdout=True, capture_stderr=True)
                        
                        # 验证生成的片段
                        if self._validate_output_clip(clip_path):
                            clip_paths.append(clip_path)
                            self.logger.info(f"Saved individual clip: {clip_path}")
                        else:
                            self.logger.warning(f"Generated clip is invalid: {clip_path}")
                            
                    except ffmpeg.Error as e:
                        self.logger.error(f"FFmpeg error for segment {i}: {e.stderr.decode('utf8')}")
                        continue
                    except Exception as e:
                        self.logger.error(f"Error processing segment {i}: {str(e)}")
                        continue
                
                if not clip_paths:
                    raise ValueError("No valid video clips were generated")
                
                self.logger.debug(f"Number of clips generated: {len(clip_paths)}")
                return clip_paths
                
            except ffmpeg.Error as e:
                self.logger.error(f"Error probing video: {e.stderr.decode('utf8')}")
                raise
                
        except Exception as e:
            self.logger.error(f"Error generating video clips: {str(e)}")
            raise

    def _parse_segments(self, segments: Union[str, dict, list]) -> list:
        """解析并验证时间段信息"""
        if isinstance(segments, str):
            try:
                for encoding in ['utf-8', 'utf-8-sig', 'gbk', 'shift-jis', 'euc-jp']:
                    try:
                        decoded_str = segments.encode(encoding).decode(encoding)
                        segments = json.loads(decoded_str)
                        self.logger.info(f"Successfully parsed JSON with {encoding} encoding")
                        break
                    except (UnicodeError, json.JSONDecodeError):
                        continue
                else:
                    raise ValueError("Failed to parse JSON with any known encoding")
            except Exception as e:
                self.logger.error(f"Failed to parse segments: {str(e)}")
                raise ValueError(f"Failed to parse segments: {str(e)}")
        
        # 统一格式为列表
        if isinstance(segments, dict):
            segments = [segments]
        elif not isinstance(segments, list):
            raise ValueError(f"segments must be a string, dict, or list, but got {type(segments)}")
            
        return segments

    def _validate_segment(self, segment: dict, total_duration: float) -> bool:
        """验证时间段是否有效"""
        if 'start' not in segment or 'end' not in segment:
            self.logger.warning(f"Invalid segment format: {segment}")
            return False
            
        start_time = segment['start']
        end_time = segment['end']
        
        if not (isinstance(start_time, (int, float)) and isinstance(end_time, (int, float))):
            self.logger.warning(f"Invalid time format: start={start_time}, end={end_time}")
            return False
            
        if not (0 <= start_time < end_time <= total_duration):
            self.logger.warning(f"Invalid time range: {start_time}-{end_time}")
            return False
            
        return True

    def _validate_output_clip(self, clip_path: str) -> bool:
        """验证输出的视频片段是否有效"""
        if not os.path.exists(clip_path) or os.path.getsize(clip_path) == 0:
            return False
            
        try:
            probe = ffmpeg.probe(clip_path)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            duration = float(probe['format']['duration'])
            return duration > 0 and int(video_info.get('nb_frames', 0)) > 0
        except Exception:
            return False

    def _get_optimal_encoding_params(self) -> Dict[str, Any]:
        """获取最优编码参数"""
        try:
            # 检查是否支持 NVIDIA GPU 编码
            result = ffmpeg.probe(None, extra_args=['-encoders'])
            if 'h264_nvenc' in result:
                return {
                    'c:v': 'h264_nvenc',  # NVIDIA GPU 加速
                    'preset': 'p4',
                    'tune': 'hq',
                    'b:v': '2000k',
                    'maxrate': '4000k',
                    'bufsize': '8000k',
                }
        except Exception:
            pass
            
        # 回退到 CPU 编码
        return {
            'c:v': 'libx264',
            'preset': 'medium',
            'crf': '23',
            'b:v': '2000k',
        }



    def update_video_narration(self, raw_merged_video_path, narration, preset):
        """
        更新视频的旁白内容
        
        Args:
            raw_merged_video_path (str): 合并后的原始视频路径
            narration (str): 原始旁白内容
            preset (dict): 预设配置
            
        Returns:
            dict: 包含更新后旁白的字典
            
        Raises:
            Exception: 当旁白生成失败时抛出异常
        """
        try:
            # 生成旁白
            narration_result = {
                "narration": ""
            }
            narration_result['narration'] = self.generate_narration(raw_merged_video_path, narration, preset)

            return narration_result
        
        except Exception as e:
            self.logger.error(f"Error updating video narration: {str(e)}")
            raise

        

    def generate_video_metadata(self, selected_clips, preset):
        """
        生成视频的元数据（标题、描述和旁白）
        
        Args:
            raw_merged_video_path (str): 合并后的原始视频路径
            selected_clip_transcription (dict): 选中的视频片段转写内容
            preset (dict): 预设配置
            
        Returns:
            dict: 包含标题、描述和旁白的字典
        """
        try:
            post_result = {
                "title": "",
                "description": ""
            }
            # 生成标题
            post_result['title'] = self.generate_title(selected_clips, preset)

            # 生成描述
            post_result['description'] = self.generate_description(selected_clips, preset)

            
            return post_result
        
        except Exception as e:
            self.logger.error(f"Error generating video metadata: {str(e)}")
            raise
        

    def _parse_transcript(self, transcript, chunk_start_ms):
        lines = transcript.split('\n')
        segments = []
        current_segment = None

        for line in lines:
            if re.match(r'\d+$', line):  # 序号行，忽略
                continue
            elif '-->' in line:  # 时间戳行
                if current_segment:
                    segments.append(current_segment)
                start, end = self._parse_timestamp(line, chunk_start_ms)
                current_segment = {'start': start, 'end': end, 'text': ''}
            elif line.strip():  # 文本行
                if current_segment:
                    current_segment['text'] += line.strip() + ' '

        if current_segment:  # 添加最后一个段落
            segments.append(current_segment)

        return segments

    def _parse_timestamp(self, timestamp_line, chunk_start_ms):
        start, end = timestamp_line.split('-->')
        start = self._time_to_ms(start.strip()) + chunk_start_ms
        end = self._time_to_ms(end.strip()) + chunk_start_ms
        return start / 1000, end / 1000  # 转换为秒

    def _time_to_ms(self, time_str):
        h, m, s = time_str.split(':')
        s, ms = s.split(',')
        return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)
    
    def _get_watermark_position(self, position):
        positions = {
            1: ('left', 'top'),
            2: ('right', 'top'),
            3: ('left', 'bottom'),
            4: ('right', 'bottom'),
            5: ('center', 'center')
        }
        return positions.get(position, ('right', 'bottom'))
    
    '''
    0=无
    1=欢快
    2=旅途
    3=忧伤
    4=安静
    5=热烈
    '''
    def _get_bgm_path(self, bgm_type):
        # Define a mapping of bgm types to file paths
        bgm_paths = {
            1: os.path.join(os.getcwd(), 'music', 'happy.MP3'),
            2: os.path.join(os.getcwd(), 'music', 'journey.MP3'),
            3: os.path.join(os.getcwd(), 'music', 'sad.MP3'),
            4: os.path.join(os.getcwd(), 'music', 'quiet.MP3'),
            5: os.path.join(os.getcwd(), 'music', 'warm.MP3')
        }
        return bgm_paths.get(bgm_type, os.path.join(os.getcwd(),'music','quiet.mp3'))
    

    def _clean_title(self, title, lang):
        """
        清理标题文本
        """
        # 去除可能的引号和多余空格
        title = title.strip(' \'"')
        
        # 根据语言进行特定处理
        if lang == 'zh':
            # 移除中文标点
            title = re.sub(r'[，。！？、：；]', '', title)
        elif lang == 'en':
            # 确保英文标题大小写正确
            title = title.title()
            # 移除英文标点
            title = re.sub(r'[,.!?:;]', '', title)
        elif lang == 'ja':
            # 移除日文标点
            title = re.sub(r'[、。！？：；]', '', title)
            
        return title.strip()


    def _calculate_text_position(self, position, frame_size, text_width, text_height):
        """
        计算文本位置
        
        Args:
            position (str): 位置标识
            frame_size (tuple): 帧大小 (width, height)
            text_width (int): 文本宽度
            text_height (int): 文本高度
            
        Returns:
            tuple: (x, y) 坐标
        """
        padding = 20  # 边距
        
        # 位置映射字典
        positions = {
            'diagonal-center': (
                (frame_size[0] - text_width) // 2 + frame_size[0] // 8,
                (frame_size[1] - text_height) // 2 + frame_size[1] // 8
            ),
            'top-left': (
                padding,
                padding + text_height
            ),
            'top-right': (
                frame_size[0] - text_width - padding,
                padding + text_height
            ),
            'center': (
                (frame_size[0] - text_width) // 2,
                (frame_size[1] - text_height) // 2
            ),
            'bottom-left': (
                padding,
                frame_size[1] - text_height - padding
            ),
            'bottom-right': (
                frame_size[0] - text_width - padding,
                frame_size[1] - text_height - padding
            ),
            'bottom': (
                (frame_size[0] - text_width) // 2,
                frame_size[1] - text_height - padding * 6  # 增加了padding的倍数，使文字位置更靠上
            ),
            'top': (
                (frame_size[0] - text_width) // 2,
                padding + text_height
            )
        }
        
        # 获取位置，默认为底部居中
        return positions.get(position, positions['bottom'])
    
    

    def _extract_tags(self, tags_text):
        """从文本中提取标签列表"""
        try:
            # 移除常见的标签前缀
            clean_text = tags_text.replace('标签：', '').replace('Tags:', '').replace('タグ：', '')
            
            # 分割文本获取标签
            tags = []
            for tag in re.split(r'[,，、\n]', clean_text):
                tag = tag.strip().strip('#').strip()
                if tag and len(tag) > 0:
                    tags.append(tag)
            
            return list(set(tags))  # 去重
        except Exception as e:
            self.logger.error(f"Error extracting tags: {str(e)}")
            return []

    def _calculate_confidence(self, description, tags):
        """计算分析结果的置信度"""
        try:
            # 基于描述长度和标签数量计算一个简单的置信度分数
            description_score = min(len(description) / 200, 1.0)  # 假设理想描述长度为200字符
            tags_score = min(len(tags) / 5, 1.0)  # 假设理想标签数量为5个
            
            # 综合评分
            confidence = (description_score * 0.7 + tags_score * 0.3)
            
            return round(confidence, 2)
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {str(e)}")
            return 0.0
        
    
    def _select_voice_id(self, voice_map, narration_voice, narration_style):
        """
        根据旁白声音和风格选择合适的voice_id
        
        narration_voice: 0=无, 1=男性, 2=女性
        narration_style: 0=无, 1=慈祥, 2=文静, 3=幽默, 4=雄壮, 5=和蔼
        """
        if narration_voice == 0 or narration_style == 0:
            return None
            
        # 定义风格到声音特征的映射
        style_to_description = {
            1: ['warm', 'friendly'],  # 慈祥
            2: ['soft', 'articulate'],  # 文静
            3: ['casual', 'expressive'],  # 幽默
            4: ['intense', 'deep'],  # 雄壮
            5: ['friendly', 'warm']  # 和蔼
        }
        
        # 获取当前风格对应的描述词列表
        style_descriptions = style_to_description.get(narration_style, [])
        
        # 筛选符合性别和风格的声音
        suitable_voices = []

        for name, voice in voice_map.items():
            labels = voice.get('labels', {})
            
            # 检查性别匹配
            gender_match = (
                (narration_voice == 1 and labels.get('gender') == 'male') or
                (narration_voice == 2 and labels.get('gender') == 'female')
            )
            
            if not gender_match:
                continue
                
            # 检查风格匹配
            description = labels.get('description', '').lower()
            if any(style in description for style in style_descriptions):
                suitable_voices.append(voice['id'])
        
        # 如果找到合适的声音，返回第一个；否则返回该性别的任意声音
        if suitable_voices:
            return suitable_voices[0]
        
        # 备选：返回符合性别的第一个声音
        for voice in voice_map.values():
            labels = voice.get('labels', {})
            if ((narration_voice == 1 and labels.get('gender') == 'male') or
                (narration_voice == 2 and labels.get('gender') == 'female')):
                return voice['id']
        
        return None


    
    def _normalize_narration_punctuation(self, narration, lang):
        """
        规范化旁白文本的标点符号和格式
        
        Args:
            narration (str): 原始旁白文本
            lang (str): 语言代码 ('zh', 'en', 'ja')
            
        Returns:
            str: 格式化后的旁白文本
        """
        # 去除多余的空格和换行
        narration = re.sub(r'\s+', ' ', narration).strip()
        narration = narration.strip('"""\'\'\'')  # 移除可能的引号
        
        # 根据语言进行特定处理
        if lang == 'zh':
            # 确保中文标点符号正确使用
            narration = re.sub(r'[!?]', '！', narration)  # 统一感叹号
            narration = re.sub(r'\.{3,}', '……', narration)  # 统一省略号
        elif lang == 'en':
            # 确保英文标点符号正确使用
            narration = re.sub(r'\.{3,}', '...', narration)  # 统一省略号
            # 确保句子首字母大写
            narration = '. '.join(s.capitalize() for s in narration.split('. '))
        elif lang == 'ja':
            # 确保日文标点符号正确使用
            narration = re.sub(r'!', '！', narration)  # 统一感叹号
            narration = re.sub(r'\.{3,}', '……', narration)  # 统一省略号
        
        return narration.strip()

    
    