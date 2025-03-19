import subprocess
import logging
from typing import Optional, Dict

class GPUUtil:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._gpu_available = None  # 缓存检查结果

    def check_gpu_available(self) -> bool:
        """
        检查是否有可用的 NVIDIA GPU
        Returns:
            bool: True 如果有可用的 GPU，否则 False
        """
        # 如果已经检查过，直接返回缓存的结果
        if self._gpu_available is not None:
            return self._gpu_available

        try:
            # 检查 nvidia-smi 命令是否可用
            result = subprocess.run(
                ['nvidia-smi'], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                timeout=5  # 添加超时限制
            )
            self._gpu_available = result.returncode == 0
            if self._gpu_available:
                self.logger.info("NVIDIA GPU is available")
            else:
                self.logger.warning("NVIDIA GPU is not available")
            return self._gpu_available
        except (FileNotFoundError, subprocess.TimeoutError) as e:
            self.logger.warning(f"Failed to check GPU availability: {str(e)}")
            self._gpu_available = False
            return False

    def get_gpu_info(self) -> Optional[Dict]:
        """
        获取 GPU 详细信息
        Returns:
            Optional[Dict]: GPU 信息字典，如果没有 GPU 则返回 None
        """
        if not self.check_gpu_available():
            return None

        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=gpu_name,memory.total,memory.free,temperature.gpu', 
                 '--format=csv,noheader,nounits'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
                text=True
            )
            
            if result.returncode == 0:
                # 解析输出
                gpu_info = result.stdout.strip().split(',')
                return {
                    'name': gpu_info[0].strip(),
                    'total_memory': float(gpu_info[1].strip()),
                    'free_memory': float(gpu_info[2].strip()),
                    'temperature': float(gpu_info[3].strip())
                }
            return None
        except Exception as e:
            self.logger.error(f"Error getting GPU info: {str(e)}")
            return None

    def get_encoding_params(self) -> Dict:
        """
        根据 GPU 可用性返回适当的编码参数
        Returns:
            Dict: 编码参数字典
        """
        if self.check_gpu_available():
            return {
                'vcodec': 'h264_nvenc',
                'acodec': 'aac',
                'preset': 'p4',
                'pix_fmt': 'yuv420p',
                'gpu': '0',
                'rc:v': 'cbr',
                'b:v': '5M'
            }
        else:
            return {
                'vcodec': 'libx264',
                'acodec': 'aac',
                'preset': 'medium',
                'pix_fmt': 'yuv420p'
            }