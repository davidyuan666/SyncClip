"""
Cloud dependency fallback utilities.
Makes COS, ElevenLabs TTS, and remote LLM optional with local fallbacks.
"""
import functools
import logging
import os
import warnings
from typing import Callable, Any, Optional

logger = logging.getLogger(__name__)

_COS_AVAILABLE = os.getenv("COS_ENABLED", "0").lower() in ("1", "true", "yes")
_TTS_AVAILABLE = os.getenv("TTS_ENABLED", "0").lower() in ("1", "true", "yes")
_LLM_REMOTE_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))


def is_cos_available() -> bool:
    return _COS_AVAILABLE

def is_tts_available() -> bool:
    return _TTS_AVAILABLE

def is_remote_llm_available() -> bool:
    return _LLM_REMOTE_AVAILABLE


def cloud_optional(fallback_value: Any = None, service: str = ""):
    """
    Decorator that returns fallback_value if cloud service is unavailable.

    Usage:
        @cloud_optional(fallback_value=[], service="cos")
        def upload_file(self, path, cos_path):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if service == "cos" and not _COS_AVAILABLE:
                logger.debug(f"[cloud_fallback] COS disabled, returning fallback for {func.__name__}")
                return fallback_value
            if service == "tts" and not _TTS_AVAILABLE:
                logger.debug(f"[cloud_fallback] TTS disabled, returning fallback for {func.__name__}")
                return fallback_value
            if service == "llm" and not _LLM_REMOTE_AVAILABLE:
                logger.debug(f"[cloud_fallback] Remote LLM disabled, returning fallback for {func.__name__}")
                return fallback_value
            return func(*args, **kwargs)
        return wrapper
    return decorator


class LocalFallbackStore:
    """Local file-based fallback for COS operations."""

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or os.path.join(os.getcwd(), "temp", "local_store")
        os.makedirs(self.base_dir, exist_ok=True)

    def download(self, remote_path: str, local_path: str) -> str:
        local_fallback = os.path.join(self.base_dir, os.path.basename(remote_path))
        if os.path.exists(local_fallback):
            logger.info(f"[local_fallback] Using cached file: {local_fallback}")
            return local_fallback
        if os.path.exists(remote_path):
            import shutil
            shutil.copy2(remote_path, local_path)
            return local_path
        raise FileNotFoundError(f"File not found: {remote_path}")

    def upload(self, local_path: str, remote_path: str) -> str:
        import shutil
        dest = os.path.join(self.base_dir, os.path.basename(remote_path))
        shutil.copy2(local_path, dest)
        return dest

    def get_url(self, remote_path: str) -> str:
        local = os.path.join(self.base_dir, os.path.basename(remote_path))
        if os.path.exists(local):
            return f"file:///{local.replace(os.sep, '/')}"
        return remote_path


class MockTTS:
    """Mock TTS that generates silence audio for testing."""

    @staticmethod
    def generate(text: str, output_path: str, duration_per_char: float = 0.08) -> str:
        import struct
        import wave
        sample_rate = 22050
        n_samples = int(len(text) * duration_per_char * sample_rate)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with wave.open(output_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(struct.pack(f"<{n_samples}h", *([0] * n_samples)))
        return output_path


def get_cloud_status() -> dict:
    return {
        "cos_available": _COS_AVAILABLE,
        "tts_available": _TTS_AVAILABLE,
        "remote_llm_available": _LLM_REMOTE_AVAILABLE,
        "openai_key_set": bool(os.getenv("OPENAI_API_KEY")),
        "cos_bucket": os.getenv("COS_BUCKET_NAME", "not_set"),
        "base_cos_url": os.getenv("BASE_COS_URL", "not_set"),
    }
