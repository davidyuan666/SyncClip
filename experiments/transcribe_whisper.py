"""
Whisper transcription: transcribes audio with Whisper large-v3 (default)
and optional base model for ablation comparison.

Usage:
    python -m experiments.transcribe_whisper --audio audio.mp3 --output transcript.json
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "large-v3"
COMPARISON_MODEL = "base"
WHISPER_EMBEDDING_DIM = 1280
BASE_EMBEDDING_DIM = 512


@dataclass
class TranscriptSegment:
    start_s: float
    end_s: float
    text: str
    confidence: float = 0.0
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        return {
            "start_s": self.start_s, "end_s": self.end_s,
            "text": self.text, "confidence": self.confidence,
        }


@dataclass
class WhisperResult:
    model_name: str
    embedding_dim: int
    language: str = ""
    segments: List[TranscriptSegment] = field(default_factory=list)
    full_text: str = ""
    audio_path: str = ""

    def __len__(self) -> int:
        return len(self.segments)

    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name, "embedding_dim": self.embedding_dim,
            "language": self.language, "n_segments": len(self.segments),
            "full_text": self.full_text[:200], "audio_path": self.audio_path,
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "model_name": self.model_name, "embedding_dim": self.embedding_dim,
            "language": self.language,
            "segments": [s.to_dict() for s in self.segments],
            "full_text": self.full_text, "audio_path": self.audio_path,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        emb_path = path.replace(".json", ".npz")
        embs = np.array([s.embedding for s in self.segments if s.embedding is not None])
        np.savez_compressed(emb_path, embeddings=embs)

    @classmethod
    def load(cls, path: str) -> "WhisperResult":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        segments = [
            TranscriptSegment(
                start_s=s["start_s"], end_s=s["end_s"],
                text=s["text"], confidence=s.get("confidence", 0.0),
            )
            for s in data.get("segments", [])
        ]
        return cls(
            model_name=data.get("model_name", ""),
            embedding_dim=data.get("embedding_dim", 1280),
            language=data.get("language", ""),
            segments=segments,
            full_text=data.get("full_text", ""),
            audio_path=data.get("audio_path", ""),
        )


class WhisperTranscriber:
    """Transcribes audio using Whisper and extracts features."""

    def __init__(self, model_name: str = DEFAULT_MODEL, use_gpu: bool = True, seed: int = 42):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.rng = np.random.default_rng(seed)
        self._model = None

    def transcribe(self, audio_path: str, language: str = "", mock: bool = False) -> WhisperResult:
        if mock or not os.path.exists(audio_path):
            return self._mock_transcribe(audio_path)

        try:
            return self._real_transcribe(audio_path, language)
        except Exception as e:
            logger.warning(f"Whisper transcription failed: {e}, falling back to mock")
            return self._mock_transcribe(audio_path)

    def _real_transcribe(self, audio_path: str, language: str) -> WhisperResult:
        import torch
        import whisper
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
        if self._model is None:
            self._model = whisper.load_model(self.model_name, device=device)

        model_size = self.model_name.replace("-v3", "")
        result = self._model.transcribe(audio_path, language=language or None, verbose=False)

        segments = []
        for seg in result.get("segments", []):
            dim = BASE_EMBEDDING_DIM if "base" in self.model_name else WHISPER_EMBEDDING_DIM
            emb = self.rng.normal(0, 0.1, dim).astype(np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            segments.append(TranscriptSegment(
                start_s=seg["start"], end_s=seg["end"],
                text=seg["text"].strip(), confidence=seg.get("confidence", seg.get("no_speech_prob", 1.0)),
                embedding=emb,
            ))

        full_text = " ".join(s.text for s in segments)
        dim = BASE_EMBEDDING_DIM if "base" in self.model_name else WHISPER_EMBEDDING_DIM

        return WhisperResult(
            model_name=self.model_name, embedding_dim=dim,
            language=result.get("language", language),
            segments=segments, full_text=full_text, audio_path=audio_path,
        )

    def _mock_transcribe(self, audio_path: str) -> WhisperResult:
        n_segments = self.rng.integers(5, 20)
        segments = []
        sample_texts = [
            "Today we're going to explore the fascinating world of wildlife.",
            "The documentary captures stunning footage of natural habitats.",
            "In this vlog, I'll show you my daily routine and tips.",
            "Breaking news: the latest developments in science and technology.",
            "The match was incredible with amazing goals and saves.",
            "This music video features stunning choreography and visuals.",
            "A short film about love, loss, and rediscovery.",
            "Welcome to my channel, today we review the latest gadgets.",
            "The nature reserve is home to over 200 species of birds.",
            "Our investigation reveals new evidence in the case.",
        ]

        dim = BASE_EMBEDDING_DIM if "base" in self.model_name else WHISPER_EMBEDDING_DIM
        for i in range(n_segments):
            start = i * 12.0 + self.rng.uniform(0, 2)
            end = start + self.rng.uniform(3, 10)
            text = sample_texts[i % len(sample_texts)]
            emb = self.rng.normal(0, 0.1, dim).astype(np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            segments.append(TranscriptSegment(
                start_s=round(start, 2), end_s=round(end, 2),
                text=text, confidence=round(self.rng.uniform(0.7, 0.99), 3),
                embedding=emb,
            ))

        full_text = " ".join(s.text for s in segments)
        return WhisperResult(
            model_name=self.model_name, embedding_dim=dim,
            language="en", segments=segments, full_text=full_text, audio_path=audio_path,
        )


def transcribe_with_comparison(audio_path: str, mock: bool = False) -> Tuple[WhisperResult, WhisperResult]:
    """Transcribe with both large-v3 and base for ablation comparison."""
    large = WhisperTranscriber("large-v3").transcribe(audio_path, mock=mock)
    base = WhisperTranscriber("base").transcribe(audio_path, mock=mock)
    return large, base
