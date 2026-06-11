"""
Preprocessing pipeline: frame extraction, audio extraction, timestamp generation.

Extracts frames at configurable fps (1/2/3/5), separates audio stream, and
saves all metadata with timestamps for reproducible experiments.

Usage:
    python -m experiments.preprocess --video video.mp4 --fps 5 --output-dir ./work
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

FPS_OPTIONS = [1, 2, 3, 5]


@dataclass
class VideoMetadata:
    path: str
    duration_s: float = 0.0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    has_audio: bool = False
    audio_sample_rate: int = 0
    audio_channels: int = 0
    file_size_bytes: int = 0

    def to_dict(self) -> Dict:
        return {
            "path": self.path, "duration_s": self.duration_s, "fps": self.fps,
            "resolution": f"{self.width}x{self.height}",
            "has_audio": self.has_audio,
            "audio_sample_rate": self.audio_sample_rate,
            "audio_channels": self.audio_channels,
        }


@dataclass
class Keyframe:
    frame_idx: int
    timestamp_s: float
    path: str
    features: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        return {"frame_idx": self.frame_idx, "timestamp_s": self.timestamp_s, "path": self.path}


@dataclass
class AudioSegment:
    start_s: float
    end_s: float
    text: str = ""
    confidence: float = 0.0

    def to_dict(self) -> Dict:
        return {"start_s": self.start_s, "end_s": self.end_s, "text": self.text, "confidence": self.confidence}


@dataclass
class PreprocessResult:
    video_id: str
    video_path: str
    metadata: VideoMetadata
    keyframes: Dict[int, List[Keyframe]] = field(default_factory=dict)
    audio_path: Optional[str] = None
    audio_segments: List[AudioSegment] = field(default_factory=list)
    work_dir: str = ""

    def to_dict(self) -> Dict:
        return {
            "video_id": self.video_id,
            "video_path": self.video_path,
            "metadata": self.metadata.to_dict(),
            "num_keyframe_sets": len(self.keyframes),
            "fps_options": sorted(self.keyframes.keys()),
            "audio_path": self.audio_path,
            "num_audio_segments": len(self.audio_segments),
            "work_dir": self.work_dir,
        }


class VideoPreprocessor:
    """Extracts frames at multiple fps rates, audio, and metadata from a video."""

    def __init__(self, work_dir: str = "experiments/output/work", seed: int = 42):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(seed)

    def process(self, video_path: str, video_id: str = "", fps_options: Optional[List[int]] = None,
                extract_audio: bool = True, mock: bool = False) -> PreprocessResult:
        fps_options = fps_options or FPS_OPTIONS
        video_id = video_id or Path(video_path).stem

        if mock:
            return self._mock_process(video_path, video_id, fps_options)

        metadata = self._probe_video(video_path)

        keyframes: Dict[int, List[Keyframe]] = {}
        video_work = self.work_dir / video_id
        video_work.mkdir(parents=True, exist_ok=True)

        for fps in fps_options:
            frames = self._extract_frames(video_path, video_work, fps, metadata.duration_s)
            keyframes[fps] = frames

        audio_path: Optional[str] = None
        if extract_audio and metadata.has_audio:
            audio_path = self._extract_audio(video_path, video_work)

        result = PreprocessResult(
            video_id=video_id, video_path=video_path, metadata=metadata,
            keyframes=keyframes, audio_path=audio_path, work_dir=str(video_work),
        )

        self._save_result(result)
        return result

    def _mock_process(self, video_path: str, video_id: str, fps_options: List[int]) -> PreprocessResult:
        """Generate mock preprocessing for testing without real video."""

        class _Meta:
            duration_s = 300.0
            fps = 30.0
            width = 1920
            height = 1080
            has_audio = True
            audio_sample_rate = 16000
            audio_channels = 2

        metadata = VideoMetadata(
            path=video_path, duration_s=300.0, fps=30.0,
            width=1920, height=1080, has_audio=True,
            audio_sample_rate=16000, audio_channels=2,
        )

        keyframes: Dict[int, List[Keyframe]] = {}
        video_work = self.work_dir / video_id
        video_work.mkdir(parents=True, exist_ok=True)

        for fps in fps_options:
            interval = 1.0 / fps
            n_frames = int(metadata.duration_s / interval)
            frames = []
            for i in range(n_frames):
                ts = round(i * interval, 3)
                frames.append(Keyframe(
                    frame_idx=i, timestamp_s=ts,
                    path=str(video_work / f"frame_{fps}fps_{i:05d}.jpg"),
                ))
            keyframes[fps] = frames

        result = PreprocessResult(
            video_id=video_id, video_path=video_path, metadata=metadata,
            keyframes=keyframes,
            audio_path=str(video_work / "audio.mp3"),
            work_dir=str(video_work),
        )
        self._save_result(result)
        return result

    def _probe_video(self, video_path: str) -> VideoMetadata:
        try:
            import ffmpeg
            probe = ffmpeg.probe(video_path)
            video_stream = next((s for s in probe["streams"] if s["codec_type"] == "video"), {})
            audio_stream = next((s for s in probe["streams"] if s["codec_type"] == "audio"), {})

            return VideoMetadata(
                path=video_path,
                duration_s=float(video_stream.get("duration", probe.get("format", {}).get("duration", 0))),
                fps=eval(str(video_stream.get("r_frame_rate", "30/1"))),
                width=int(video_stream.get("width", 0)),
                height=int(video_stream.get("height", 0)),
                has_audio=bool(audio_stream),
                audio_sample_rate=int(audio_stream.get("sample_rate", 0)),
                audio_channels=int(audio_stream.get("channels", 0)),
            )
        except Exception:
            try:
                cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", video_path]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                probe = json.loads(result.stdout)
                fmt = probe.get("format", {})
                return VideoMetadata(
                    path=video_path,
                    duration_s=float(fmt.get("duration", 0)),
                    fps=30.0, width=1920, height=1080,
                    has_audio=True, audio_sample_rate=44100, audio_channels=2,
                )
            except Exception:
                return VideoMetadata(path=video_path)

    def _extract_frames(self, video_path: str, work_dir: Path, fps: int, duration_s: float) -> List[Keyframe]:
        frames_dir = work_dir / f"frames_{fps}fps"
        frames_dir.mkdir(parents=True, exist_ok=True)

        try:
            import ffmpeg
            out_pattern = str(frames_dir / f"frame_%05d.jpg")
            stream = ffmpeg.input(video_path)
            stream = ffmpeg.filter(stream, "fps", fps=fps)
            stream = ffmpeg.output(stream, out_pattern, start_number=0)
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True, quiet=True)
        except Exception:
            cmd = ["ffmpeg", "-i", video_path, "-vf", f"fps={fps}", "-q:v", "2",
                   str(frames_dir / "frame_%05d.jpg"), "-y"]
            subprocess.run(cmd, capture_output=True, timeout=300)

        frame_files = sorted(frames_dir.glob("frame_*.jpg"))
        keyframes = []
        interval = 1.0 / fps
        for i, fp in enumerate(frame_files):
            keyframes.append(Keyframe(frame_idx=i, timestamp_s=round(i * interval, 3), path=str(fp)))
        return keyframes

    def _extract_audio(self, video_path: str, work_dir: Path) -> str:
        audio_path = str(work_dir / "audio.mp3")
        try:
            import ffmpeg
            stream = ffmpeg.input(video_path)
            stream = ffmpeg.output(stream.audio, audio_path, acodec="libmp3lame", ac=1, ar="16000")
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True, quiet=True)
        except Exception:
            cmd = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "libmp3lame", "-ac", "1", "-ar", "16000", audio_path, "-y"]
            subprocess.run(cmd, capture_output=True, timeout=120)
        return audio_path

    def _save_result(self, result: PreprocessResult) -> None:
        save_path = Path(result.work_dir) / "preprocess_result.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--fps", type=int, nargs="+", default=[1, 2, 3, 5])
    parser.add_argument("--output-dir", default="experiments/output/work")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    preprocessor = VideoPreprocessor(args.output_dir)
    result = preprocessor.process(args.video, fps_options=args.fps, mock=args.mock)
    print(json.dumps(result.to_dict(), indent=2))
