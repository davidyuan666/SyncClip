"""
Build segment candidates from preprocessed keyframes, CLIP features, and
Whisper transcriptions. Groups adjacent keyframes into visual segments,
attaches audio spans, and computes importance scores.

Usage:
    python -m experiments.build_candidates --preprocess result.json --clip features.npz --whisper transcript.json
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


@dataclass
class SegmentCandidate:
    segment_id: str
    start_s: float
    end_s: float
    duration_s: float
    visual_summary: str = ""
    frame_indices: List[int] = field(default_factory=list)
    keyframe_paths: List[str] = field(default_factory=list)
    visual_embedding: Optional[np.ndarray] = None
    transcript_spans: List[dict] = field(default_factory=list)
    audio_embedding: Optional[np.ndarray] = None
    importance_score: float = 0.0
    visual_change_score: float = 0.0
    request_similarity: float = 0.0
    speech_presence: bool = False
    event_presence: bool = False

    def to_dict(self) -> Dict:
        return {
            "segment_id": self.segment_id, "start_s": self.start_s, "end_s": self.end_s,
            "duration_s": self.duration_s, "visual_summary": self.visual_summary[:100],
            "n_keyframes": len(self.frame_indices),
            "importance_score": round(self.importance_score, 4),
            "visual_change_score": round(self.visual_change_score, 4),
            "request_similarity": round(self.request_similarity, 4),
            "speech_presence": self.speech_presence,
            "event_presence": self.event_presence,
        }


@dataclass
class CandidateSet:
    fps: int
    segments: List[SegmentCandidate] = field(default_factory=list)
    total_duration_s: float = 0.0

    def __len__(self) -> int:
        return len(self.segments)

    def to_dict(self) -> Dict:
        return {
            "fps": self.fps, "n_segments": len(self.segments),
            "total_duration_s": self.total_duration_s,
            "segments": [s.to_dict() for s in self.segments],
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


class CandidateBuilder:
    """
    Groups keyframes into visual segments and computes importance scores.

    Implements the paper's greedy ranking approach for segment selection.
    Defaults match Table 1 in the paper: theta=0.18, delta=0.50s.
    """

    def __init__(self, theta: float = 0.18, delta_min_gap: float = 0.50,
                 min_segment_duration: float = 1.0, seed: int = 42):
        self.theta = theta
        self.delta_min_gap = delta_min_gap
        self.min_segment_duration = min_segment_duration
        self.rng = np.random.default_rng(seed)

    def build(
        self,
        clip_features: "CLIPFeatures",
        whisper_result: Optional["WhisperResult"] = None,
        visual_summaries: Optional[List[str]] = None,
        mock: bool = False,
    ) -> CandidateSet:
        """
        Group keyframes into candidate visual segments.

        Args:
            clip_features: CLIPFeatures with embeddings and timestamps
            whisper_result: Whisper transcription result
            visual_summaries: optional BLIP2 captions per frame
        """
        fps = clip_features.fps
        embeddings = clip_features.embeddings
        timestamps = clip_features.timestamps
        n = len(timestamps)

        if embeddings is None or len(embeddings) == 0:
            return CandidateSet(fps=fps)

        # Step 1: compute visual change scores between consecutive frames
        change_scores = np.zeros(n)
        for i in range(1, n):
            if i < len(embeddings) and (i - 1) < len(embeddings):
                sim = float(np.dot(embeddings[i], embeddings[i - 1]))
                change_scores[i] = 1.0 - max(0, min(1, sim))

        # Step 2: segment boundaries where change > theta
        boundaries = [0]
        for i in range(1, n):
            if change_scores[i] > self.theta:
                if i - boundaries[-1] >= max(1, self.min_segment_duration * fps):
                    boundaries.append(i)

        if boundaries[-1] != n:
            boundaries.append(n)

        # Step 3: build segments
        segments = []
        audio_spans = whisper_result.segments if whisper_result else []

        for seg_idx in range(len(boundaries) - 1):
            i_start = boundaries[seg_idx]
            i_end = boundaries[seg_idx + 1]

            if i_end <= i_start:
                continue

            start_s = timestamps[i_start] if i_start < len(timestamps) else i_start / fps
            end_s = timestamps[i_end - 1] if (i_end - 1) < len(timestamps) else i_end / fps
            end_s = max(end_s, start_s + (i_end - i_start) / fps)
            duration = end_s - start_s

            if duration < self.min_segment_duration:
                continue

            frame_indices = list(range(i_start, i_end))
            keyframe_paths = clip_features.frame_paths[i_start:i_end]

            # Visual change score for the segment
            seg_change = float(np.mean(change_scores[i_start:i_end])) if i_end > i_start else 0.0

            # Request similarity placeholder
            request_sim = float(self.rng.uniform(0.5, 0.95)) if mock else 0.7

            # Speech presence
            has_speech = any(
                s.start_s <= end_s and s.end_s >= start_s
                for s in audio_spans
            )

            # Event presence (detected via embedding variance)
            if i_start < i_end and i_end <= len(embeddings):
                seg_emb = embeddings[i_start:i_end]
                emb_var = float(np.var(seg_emb))
            else:
                emb_var = 0.0
            has_event = emb_var > 0.05

            # Importance score: w_i = 0.5*r_i + 0.3*v_i + 0.2*a_i (paper Table 1)
            audio_cue = 0.5 * float(has_speech) + 0.5 * float(has_event)
            importance = (
                0.50 * request_sim +
                0.30 * seg_change +
                0.20 * audio_cue
            )

            # Segment visual embedding (mean of constituent frames)
            if embeddings is not None and i_start < i_end and i_end <= len(embeddings):
                seg_visual_emb = np.mean(embeddings[i_start:i_end], axis=0)
                seg_visual_emb = seg_visual_emb / (np.linalg.norm(seg_visual_emb) + 1e-8)
            else:
                seg_visual_emb = self.rng.normal(0, 0.1, 768).astype(np.float32)

            summary = visual_summaries[i_start] if visual_summaries and i_start < len(visual_summaries) else ""

            # Audio embedding from overlapping transcript spans
            seg_audio_emb = self._aggregate_audio_embedding(start_s, end_s, audio_spans)

            segments.append(SegmentCandidate(
                segment_id=f"seg_{fps}fps_{seg_idx:03d}",
                start_s=round(start_s, 2), end_s=round(end_s, 2),
                duration_s=round(duration, 2),
                visual_summary=summary,
                frame_indices=frame_indices,
                keyframe_paths=keyframe_paths,
                visual_embedding=seg_visual_emb,
                transcript_spans=[
                    {"start": s.start_s, "end": s.end_s, "text": s.text}
                    for s in audio_spans
                    if s.start_s <= end_s and s.end_s >= start_s
                ],
                audio_embedding=seg_audio_emb,
                importance_score=round(float(importance), 4),
                visual_change_score=round(float(seg_change), 4),
                request_similarity=round(float(request_sim), 4),
                speech_presence=has_speech,
                event_presence=has_event,
            ))

        total_duration = sum(s.duration_s for s in segments)

        return CandidateSet(fps=fps, segments=segments, total_duration_s=round(total_duration, 2))

    def _aggregate_audio_embedding(self, start_s: float, end_s: float,
                                    audio_spans: List) -> Optional[np.ndarray]:
        overlapping = [
            s for s in audio_spans
            if hasattr(s, "embedding") and s.embedding is not None
            and s.start_s <= end_s and s.end_s >= start_s
        ]
        if not overlapping:
            return self.rng.normal(0, 0.1, 1280).astype(np.float32)
        avg = np.mean([s.embedding for s in overlapping], axis=0)
        return avg / (np.linalg.norm(avg) + 1e-8)

    def build_multiple_fps(
        self,
        clip_by_fps: Dict[int, "CLIPFeatures"],
        whisper_result: Optional["WhisperResult"] = None,
        mock: bool = False,
    ) -> Dict[int, CandidateSet]:
        return {
            fps: self.build(features, whisper_result, mock=mock)
            for fps, features in clip_by_fps.items()
        }
