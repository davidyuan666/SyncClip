"""
Plan validator: hard and soft checks before FFmpeg rendering.
Implements guidance Section 7.

Hard checks (must pass; if fail → LLM revision requested):
  1. Timestamp boundaries are legal (start < end, within source)
  2. Segment durations are positive
  3. Target timeline does not overlap illegally
  4. Total duration matches requested tolerance
  5. sync_anchor times are inside selected intervals
  6. Transition type is supported
  7. Subtitle timestamps are inside selected segments

Soft checks (warnings; recorded in validation report):
  a. Semantic correspondence score
  b. Temporal sync error
  c. Duration deviation
  d. Number of low-confidence segments

Max 2 LLM revision attempts. After that, mark as invalid and record reason.

Usage:
    python -m experiments.validate_plan --plan plan.json --candidates candidates.json
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_TRANSITIONS = {"cut", "fade", "dissolve", "wipe", "none"}
DURATION_TOLERANCE = 2.0
MIN_SEGMENT_DURATION = 0.1
MAX_SEGMENT_DURATION = 600.0


@dataclass
class ValidationReport:
    request_id: str
    passed: bool
    hard_errors: List[str] = field(default_factory=list)
    soft_warnings: List[str] = field(default_factory=list)
    semantic_score: float = 0.0
    temporal_sync_error_ms: float = 0.0
    duration_deviation_s: float = 0.0
    low_confidence_count: int = 0
    revision_attempt: int = 0
    max_revisions: int = 2

    def to_dict(self) -> Dict:
        return {
            "request_id": self.request_id, "passed": self.passed,
            "hard_errors": self.hard_errors, "soft_warnings": self.soft_warnings,
            "semantic_score": self.semantic_score,
            "temporal_sync_error_ms": self.temporal_sync_error_ms,
            "duration_deviation_s": self.duration_deviation_s,
            "low_confidence_count": self.low_confidence_count,
            "revision_attempt": self.revision_attempt,
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


class PlanValidator:
    """Validates edit decision plans before FFmpeg rendering."""

    def __init__(self, max_revisions: int = 2, duration_tolerance: float = DURATION_TOLERANCE):
        self.max_revisions = max_revisions
        self.duration_tolerance = duration_tolerance

    def validate(self, plan: "EditDecision", candidates: Optional["CandidateSet"] = None,
                 visual_embeddings: Optional[np.ndarray] = None,
                 audio_embeddings: Optional[np.ndarray] = None) -> ValidationReport:
        report = ValidationReport(
            request_id=plan.request_id,
            passed=True,
            max_revisions=self.max_revisions,
            revision_attempt=plan.revision_count,
        )

        report.hard_errors = self._hard_checks(plan)
        report.soft_warnings = self._soft_checks(plan, candidates, visual_embeddings, audio_embeddings)

        report.passed = len(report.hard_errors) == 0

        if visual_embeddings is not None and audio_embeddings is not None and len(visual_embeddings) > 0:
            from experiments.metrics.semantic_metrics import compute_semantic_correspondence
            sem = compute_semantic_correspondence(
                [visual_embeddings[0]] if visual_embeddings.ndim == 1 else [visual_embeddings[i] for i in range(min(5, len(visual_embeddings)))],
                [audio_embeddings[0]] if audio_embeddings.ndim == 1 else [audio_embeddings[i] for i in range(min(5, len(audio_embeddings)))],
            )
            report.semantic_score = sem.mean_similarity

        report.temporal_sync_error_ms = self._estimate_sync_error(plan)
        report.duration_deviation_s = round(abs(plan.total_target_duration - plan.target_duration), 2)
        report.low_confidence_count = sum(1 for s in plan.segments if s.importance < 0.4)

        return report

    def _hard_checks(self, plan: "EditDecision") -> List[str]:
        errors = []

        if not plan.segments:
            errors.append("No segments in plan")
            return errors

        if not isinstance(plan.target_duration, (int, float)) or plan.target_duration <= 0:
            errors.append(f"Invalid target duration: {plan.target_duration}")

        for i, seg in enumerate(plan.segments):
            prefix = f"Segment[{i}] ({seg.segment_id}):"

            if seg.source_start < 0 or seg.source_end < 0:
                errors.append(f"{prefix} Negative source timestamp")

            if seg.source_start >= seg.source_end:
                errors.append(f"{prefix} source_start ({seg.source_start}) >= source_end ({seg.source_end})")
            elif seg.source_end - seg.source_start < MIN_SEGMENT_DURATION:
                errors.append(f"{prefix} Duration too short: {seg.source_end - seg.source_start:.2f}s < {MIN_SEGMENT_DURATION}s")

            if seg.target_start >= seg.target_end:
                errors.append(f"{prefix} target_start >= target_end")

            if seg.transition not in SUPPORTED_TRANSITIONS:
                errors.append(f"{prefix} Unsupported transition: {seg.transition}")

            if seg.sync_anchor:
                if not (seg.source_start <= seg.sync_anchor.video_time <= seg.source_end):
                    errors.append(f"{prefix} sync_anchor.video_time not in [{seg.source_start}, {seg.source_end}]")
                if not (seg.source_start <= seg.sync_anchor.audio_time <= seg.source_end):
                    errors.append(f"{prefix} sync_anchor.audio_time not in [{seg.source_start}, {seg.source_end}]")

        for i in range(len(plan.segments) - 1):
            a = plan.segments[i]
            b = plan.segments[i + 1]
            if abs(a.target_end - b.target_start) > 0.1:
                errors.append(f"Gap between segment {i} and {i+1}: "
                              f"target_end={a.target_end:.2f}, target_start={b.target_start:.2f}")

        total_dev = abs(plan.total_target_duration - plan.target_duration)
        if total_dev > self.duration_tolerance:
            errors.append(f"Total duration deviation {total_dev:.2f}s > tolerance {self.duration_tolerance}s")

        if plan.subtitle and plan.subtitle.enabled:
            sub = plan.subtitle
            if sub.start_s < 0:
                errors.append("Subtitle start_s is negative")
            if sub.end_s > plan.total_target_duration:
                errors.append(f"Subtitle end_s ({sub.end_s}) exceeds total target duration ({plan.total_target_duration})")

        return errors

    def _soft_checks(self, plan: "EditDecision", candidates: Optional["CandidateSet"],
                     visual_embeddings: Optional[np.ndarray], audio_embeddings: Optional[np.ndarray]) -> List[str]:
        warnings = []

        total_dev = abs(plan.total_target_duration - plan.target_duration)
        if 0.5 < total_dev <= self.duration_tolerance:
            warnings.append(f"Noticeable duration deviation: {total_dev:.2f}s")

        low_conf = [s for s in plan.segments if s.importance < 0.4]
        if low_conf:
            warnings.append(f"{len(low_conf)} segments with low importance (< 0.4)")

        if candidates:
            plan_ids = {s.segment_id for s in plan.segments}
            available_ids = {s.segment_id for s in candidates.segments}
            not_found = plan_ids - available_ids
            if not_found:
                warnings.append(f"Segment IDs not found in candidates: {not_found}")

        duration_ratio = abs(plan.total_source_duration / max(plan.total_target_duration, 0.1) - 1.0)
        if duration_ratio > 0.3:
            warnings.append(f"Source/target duration ratio large: {duration_ratio:.2f}")

        return warnings

    def _estimate_sync_error(self, plan: "EditDecision") -> float:
        errors = []
        for seg in plan.segments:
            if seg.sync_anchor:
                err = abs(seg.sync_anchor.video_time - seg.sync_anchor.audio_time)
                errors.append(err)
        if not errors:
            return 0.0
        return round(float(np.mean(errors)) * 1000, 1)


def validate_and_revise(
    plan: "EditDecision",
    candidates: "CandidateSet",
    llm_planner: "LLMPlanner",
    validator: Optional[PlanValidator] = None,
    mock: bool = False,
) -> Tuple["EditDecision", ValidationReport]:
    """Run validation loop with up to 2 LLM revisions."""
    validator = validator or PlanValidator()
    report = validator.validate(plan, candidates)

    while not report.passed and plan.revision_count < validator.max_revisions:
        logger.info(f"Revision {plan.revision_count + 1}/{validator.max_revisions} for {plan.request_id}")
        plan = llm_planner.revise(plan, report.hard_errors, candidates, mock=mock)
        report = validator.validate(plan, candidates)
        report.revision_attempt = plan.revision_count

    plan.validation_passed = report.passed
    plan.validation_errors = report.hard_errors

    if not report.passed:
        logger.warning(f"Plan {plan.request_id} still invalid after {plan.revision_count} revisions: {report.hard_errors}")

    return plan, report
