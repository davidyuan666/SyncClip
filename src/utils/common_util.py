import logging
import os
import time
import json

class CommonUtil:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.project_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'projects')

    def save_segments_to_json(self, project_no, segments, video_url):
        """保存分段信息到JSON文件"""
        try:
            # 创建segments目录
            segments_dir = os.path.join(self.project_dir, 'segments')
            os.makedirs(segments_dir, exist_ok=True)
            
            # 生成文件名
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            json_file = os.path.join(segments_dir, f'{project_no}_{timestamp}_segments.json')
            
            # 准备保存数据
            save_data = {
                'project_no': project_no,
                'timestamp': timestamp,
                'video_url': video_url,
                'total_segments': len(segments),
                'segments': segments
            }
            
            # 写入JSON文件
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
                
            print(f"Saved {len(segments)} segments to JSON file: {json_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving segments to JSON: {str(e)}")


    def save_captions_to_json(self, project_no, captions_data):
        """保存字幕数据到JSON文件"""
        try:
            # 创建captions目录
            captions_dir = os.path.join(self.project_dir, 'captions')
            os.makedirs(captions_dir, exist_ok=True)
            
            # 生成文件名（使用时间戳避免重名）
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            json_file = os.path.join(captions_dir, f'{project_no}_{timestamp}_captions.json')
            
            # 准备要保存的数据
            save_data = {
                'project_no': project_no,
                'timestamp': timestamp,
                'captions': captions_data
            }
            
            # 写入JSON文件
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
                
            print(f"Saved captions to JSON file: {json_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving captions to JSON: {str(e)}")

    def save_result_to_json(self, project_no, result):
        """保存处理结果到JSON文件"""
        try:
            # 创建results目录
            results_dir = os.path.join(self.project_dir, 'results')
            os.makedirs(results_dir, exist_ok=True)
            
            # 生成文件名
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            json_file = os.path.join(results_dir, f'{project_no}_{timestamp}_result.json')
            
            # 写入JSON文件
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                
            print(f"Saved result data to: {json_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving result data: {str(e)}")
    
    