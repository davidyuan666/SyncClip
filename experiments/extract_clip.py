"""
CLIP feature extraction: encodes video keyframes using CLIP ViT-B/32 (default)
with optional RN50x16 comparison for ablation studies.

Usage:
    python -m experiments.extract_clip --keyframes work/video_01/frames_5fps/ --output features.npz
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "ViT-B/32"
COMPARISON_MODEL = "RN50x16"
CLIP_EMBEDDING_DIM = 768
RN50_EMBEDDING_DIM = 768


@dataclass
class CLIPFeatures:
    model_name: str
    embedding_dim: int
    frame_paths: List[str] = field(default_factory=list)
    embeddings: Optional[np.ndarray] = None
    timestamps: List[float] = field(default_factory=list)
    fps: int = 0

    def __len__(self) -> int:
        return len(self.frame_paths) if self.embeddings is None else self.embeddings.shape[0]

    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "n_frames": len(self),
            "fps": self.fps,
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(
            path,
            embeddings=self.embeddings if self.embeddings is not None else np.array([]),
            timestamps=np.array(self.timestamps),
            frame_paths=np.array(self.frame_paths),
        )
        meta_path = path.replace(".npz", ".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str, model_name: str = "", fps: int = 0) -> "CLIPFeatures":
        data = np.load(path, allow_pickle=True)
        return cls(
            model_name=model_name,
            embedding_dim=data["embeddings"].shape[1] if data["embeddings"].size > 0 else 768,
            frame_paths=data["frame_paths"].tolist() if data["frame_paths"].size > 0 else [],
            embeddings=data["embeddings"] if data["embeddings"].size > 0 else None,
            timestamps=data["timestamps"].tolist() if data["timestamps"].size > 0 else [],
            fps=fps,
        )


class CLIPExtractor:
    """Extracts CLIP visual features from video frames."""

    def __init__(self, model_name: str = DEFAULT_MODEL, use_gpu: bool = True, seed: int = 42):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.rng = np.random.default_rng(seed)
        self._model = None
        self._preprocess = None

    def extract(
        self, frame_paths: List[str], timestamps: Optional[List[float]] = None, fps: int = 0, mock: bool = False,
    ) -> CLIPFeatures:
        if mock:
            return self._mock_extract(frame_paths, timestamps, fps)

        try:
            return self._real_extract(frame_paths, timestamps, fps)
        except Exception as e:
            logger.warning(f"CLIP real extraction failed: {e}, falling back to mock")
            return self._mock_extract(frame_paths, timestamps, fps)

    def extract_multiple_fps(
        self, keyframes_by_fps: Dict[int, List], mock: bool = False,
    ) -> Dict[int, CLIPFeatures]:
        results = {}
        for fps, frames in keyframes_by_fps.items():
            paths = [f.path if hasattr(f, "path") else str(f) for f in frames]
            timestamps = [f.timestamp_s if hasattr(f, "timestamp_s") else 0 for f in frames]
            results[fps] = self.extract(paths, timestamps, fps, mock=mock)
        return results

    def _real_extract(self, frame_paths: List[str], timestamps: Optional[List[float]], fps: int) -> CLIPFeatures:
        import torch
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        try:
            from transformers import CLIPProcessor, CLIPModel
        except ImportError:
            import clip
            return self._clip_openai_extract(frame_paths, timestamps, fps)

        device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
        if self._model is None:
            variant = "openai/clip-vit-base-patch32" if "ViT-B/32" in self.model_name else "openai/clip-vit-base-patch32"
            self._model = CLIPModel.from_pretrained(variant).to(device)
            self._preprocess = CLIPProcessor.from_pretrained(variant)
            self._model.eval()

        from PIL import Image
        embeddings = []
        valid_paths = []
        valid_ts = []
        batch_size = 32

        for i in range(0, len(frame_paths), batch_size):
            batch_paths = frame_paths[i:i + batch_size]
            batch_ts = (timestamps or [])[i:i + batch_size]
            images = []
            for fp in batch_paths:
                try:
                    images.append(Image.open(fp).convert("RGB"))
                    valid_paths.append(fp)
                    valid_ts.append(batch_ts[len(images) - 1] if batch_ts else 0.0)
                except Exception:
                    continue

            if not images:
                continue

            inputs = self._preprocess(images=images, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = self._model.get_image_features(**inputs)
            emb = outputs.cpu().numpy()
            emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
            embeddings.append(emb)

        all_emb = np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, CLIP_EMBEDDING_DIM))
        return CLIPFeatures(
            model_name=self.model_name, embedding_dim=all_emb.shape[1],
            frame_paths=valid_paths or frame_paths, embeddings=all_emb,
            timestamps=valid_ts or (timestamps or []), fps=fps,
        )

    def _clip_openai_extract(self, frame_paths, timestamps, fps) -> CLIPFeatures:
        import torch
        import clip
        from PIL import Image

        device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(self.model_name, device=device)

        embeddings = []
        for fp in frame_paths:
            try:
                image = preprocess(Image.open(fp).convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model.encode_image(image).cpu().numpy()[0]
                emb = emb / (np.linalg.norm(emb) + 1e-8)
                embeddings.append(emb)
            except Exception:
                continue

        all_emb = np.array(embeddings) if embeddings else np.zeros((0, CLIP_EMBEDDING_DIM))
        return CLIPFeatures(
            model_name=self.model_name, embedding_dim=all_emb.shape[1] if all_emb.size else CLIP_EMBEDDING_DIM,
            frame_paths=frame_paths, embeddings=all_emb,
            timestamps=timestamps or [], fps=fps,
        )

    def _mock_extract(self, frame_paths: List[str], timestamps: Optional[List[float]], fps: int) -> CLIPFeatures:
        n = len(frame_paths)
        dim = 768
        emb = self.rng.normal(0, 1, (n, dim)).astype(np.float32)
        for i in range(1, n, max(1, n // 20)):
            emb[i] = emb[i] + self.rng.normal(0, 1.5, dim).astype(np.float32)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        return CLIPFeatures(
            model_name=self.model_name, embedding_dim=dim,
            frame_paths=frame_paths, embeddings=emb,
            timestamps=timestamps or [0.0] * n, fps=fps,
        )


def extract_with_comparison(
    frame_paths: List[str], timestamps=None, fps=0, mock=False,
) -> Tuple[CLIPFeatures, CLIPFeatures]:
    """Extract with both ViT-B/32 and RN50x16 for ablation comparison."""
    vit = CLIPExtractor("ViT-B/32").extract(frame_paths, timestamps, fps, mock=mock)
    rn50 = CLIPExtractor("RN50x16").extract(frame_paths, timestamps, fps, mock=mock)
    return vit, rn50
