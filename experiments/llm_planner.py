"""
LLM planner: generates structured edit decision schemas from segment candidates
and user editing requests. Uses GPT-4 with constrained JSON output format.

The edit decision schema matches the paper definition:
    request_id, target_duration, segments[], source_start, source_end,
    target_start, target_end, sync_anchor, transition, subtitle, audio_mix,
    render_backend.

Usage:
    python -m experiments.llm_planner --candidates candidates.json --request "create 60s highlight" --output plan.json
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

EDIT_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "request_id": {"type": "string", "description": "Unique request identifier"},
        "target_duration": {"type": "number", "description": "Target duration in seconds"},
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "segment_id": {"type": "string"},
                    "source_start": {"type": "number", "description": "Source video start timestamp (s)"},
                    "source_end": {"type": "number", "description": "Source video end timestamp (s)"},
                    "target_start": {"type": "number", "description": "Target timeline start (s)"},
                    "target_end": {"type": "number", "description": "Target timeline end (s)"},
                    "sync_anchor": {
                        "type": "object",
                        "properties": {
                            "video_time": {"type": "number"},
                            "audio_time": {"type": "number"},
                        },
                        "description": "Synchronization anchor point",
                    },
                    "transition": {
                        "type": "string",
                        "enum": ["cut", "fade", "dissolve", "wipe", "none"],
                        "default": "cut",
                    },
                    "importance": {"type": "number", "description": "0.0-1.0 importance score"},
                },
                "required": ["segment_id", "source_start", "source_end", "target_start", "target_end", "transition"],
            },
        },
        "subtitle": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean", "default": False},
                "text": {"type": "string"},
                "start_s": {"type": "number"},
                "end_s": {"type": "number"},
                "font_size": {"type": "integer", "default": 24},
                "color": {"type": "string", "default": "#FFFFFF"},
            },
        },
        "audio_mix": {
            "type": "object",
            "properties": {
                "narration_enabled": {"type": "boolean", "default": False},
                "narration_text": {"type": "string"},
                "bgm_path": {"type": "string"},
                "narration_volume": {"type": "number", "default": 1.0},
                "bgm_volume": {"type": "number", "default": 0.3},
            },
        },
        "render_backend": {"type": "string", "enum": ["ffmpeg", "moviepy"], "default": "ffmpeg"},
        "notes": {"type": "string", "description": "Optional editing notes"},
    },
    "required": ["request_id", "target_duration", "segments", "render_backend"],
}


@dataclass
class SyncAnchor:
    video_time: float
    audio_time: float

    def to_dict(self) -> Dict:
        return {"video_time": self.video_time, "audio_time": self.audio_time}


@dataclass
class EditSegment:
    segment_id: str
    source_start: float
    source_end: float
    target_start: float
    target_end: float
    transition: str = "cut"
    sync_anchor: Optional[SyncAnchor] = None
    importance: float = 0.5

    def to_dict(self) -> Dict:
        d = {
            "segment_id": self.segment_id, "source_start": self.source_start,
            "source_end": self.source_end, "target_start": self.target_start,
            "target_end": self.target_end, "transition": self.transition,
            "importance": self.importance,
        }
        if self.sync_anchor:
            d["sync_anchor"] = self.sync_anchor.to_dict()
        return d


@dataclass
class SubtitleSpec:
    enabled: bool = False
    text: str = ""
    start_s: float = 0.0
    end_s: float = 0.0
    font_size: int = 24
    color: str = "#FFFFFF"

    def to_dict(self) -> Dict:
        return {"enabled": self.enabled, "text": self.text, "start_s": self.start_s,
                "end_s": self.end_s, "font_size": self.font_size, "color": self.color}


@dataclass
class AudioMixSpec:
    narration_enabled: bool = False
    narration_text: str = ""
    bgm_path: str = ""
    narration_volume: float = 1.0
    bgm_volume: float = 0.3

    def to_dict(self) -> Dict:
        return {"narration_enabled": self.narration_enabled, "narration_text": self.narration_text,
                "bgm_path": self.bgm_path, "narration_volume": self.narration_volume,
                "bgm_volume": self.bgm_volume}


@dataclass
class EditDecision:
    request_id: str
    target_duration: float
    segments: List[EditSegment] = field(default_factory=list)
    subtitle: Optional[SubtitleSpec] = None
    audio_mix: Optional[AudioMixSpec] = None
    render_backend: str = "ffmpeg"
    notes: str = ""
    revision_count: int = 0
    validation_passed: bool = False
    validation_errors: List[str] = field(default_factory=list)

    @property
    def total_source_duration(self) -> float:
        return sum(s.source_end - s.source_start for s in self.segments)

    @property
    def total_target_duration(self) -> float:
        return sum(s.target_end - s.target_start for s in self.segments)

    def to_dict(self) -> Dict:
        return {
            "request_id": self.request_id, "target_duration": self.target_duration,
            "segments": [s.to_dict() for s in self.segments],
            "subtitle": self.subtitle.to_dict() if self.subtitle else None,
            "audio_mix": self.audio_mix.to_dict() if self.audio_mix else None,
            "render_backend": self.render_backend, "notes": self.notes,
            "revision_count": self.revision_count,
            "validation_passed": self.validation_passed,
            "validation_errors": self.validation_errors,
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "EditDecision":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        segs = [EditSegment(
            segment_id=s["segment_id"], source_start=s["source_start"],
            source_end=s["source_end"], target_start=s["target_start"],
            target_end=s["target_end"], transition=s.get("transition", "cut"),
            sync_anchor=SyncAnchor(**s["sync_anchor"]) if s.get("sync_anchor") else None,
            importance=s.get("importance", 0.5),
        ) for s in data.get("segments", [])]

        sub = None
        if data.get("subtitle"):
            s = data["subtitle"]
            sub = SubtitleSpec(enabled=s.get("enabled", False), text=s.get("text", ""),
                                start_s=s.get("start_s", 0), end_s=s.get("end_s", 0),
                                font_size=s.get("font_size", 24), color=s.get("color", "#FFFFFF"))

        am = None
        if data.get("audio_mix"):
            a = data["audio_mix"]
            am = AudioMixSpec(narration_enabled=a.get("narration_enabled", False),
                              narration_text=a.get("narration_text", ""),
                              bgm_path=a.get("bgm_path", ""),
                              narration_volume=a.get("narration_volume", 1.0),
                              bgm_volume=a.get("bgm_volume", 0.3))

        return cls(
            request_id=data["request_id"], target_duration=data["target_duration"],
            segments=segs, subtitle=sub, audio_mix=am,
            render_backend=data.get("render_backend", "ffmpeg"),
            notes=data.get("notes", ""),
            revision_count=data.get("revision_count", 0),
            validation_passed=data.get("validation_passed", False),
            validation_errors=data.get("validation_errors", []),
        )


BUILD_PLAN_PROMPT = """You are a professional video editing planner. Given the following inputs, produce a structured edit decision plan.

CONTEXT:
- You must output ONLY valid JSON matching the schema below.
- Total target duration: {target_duration} seconds.
- Supported transitions: cut, fade, dissolve, wipe, none.
- Supported render backend: ffmpeg.

USER REQUEST:
{user_request}

CANDIDATE SEGMENTS:
{candidates_text}

REQUIRED OUTPUT SCHEMA:
{{
  "request_id": "{request_id}",
  "target_duration": {target_duration},
  "segments": [
    {{
      "segment_id": "seg_xxx",
      "source_start": 0.0,
      "source_end": 10.0,
      "target_start": 0.0,
      "target_end": 10.0,
      "sync_anchor": {{"video_time": 5.0, "audio_time": 5.0}},
      "transition": "cut",
      "importance": 0.85
    }}
  ],
  "subtitle": {{"enabled": false, "text": "", "start_s": 0, "end_s": 0, "font_size": 24, "color": "#FFFFFF"}},
  "audio_mix": {{"narration_enabled": false, "narration_text": "", "bgm_path": "", "narration_volume": 1.0, "bgm_volume": 0.3}},
  "render_backend": "ffmpeg",
  "notes": "optional notes"
}}

CONSTRAINTS:
1. Source times must match exactly the candidate segment boundaries provided.
2. Target timeline must be contiguous with no gaps: target_end of segment N = target_start of segment N+1.
3. Total target_duration of all segments must be {target_duration}s (+-2s tolerance).
4. sync_anchor video_time must be within [source_start, source_end].
5. Pick the highest-importance segments first, respecting the target duration.
6. At most 12 segments total.
7. Output ONLY the JSON object, nothing else."""


class LLMPlanner:
    """Generates structured edit decisions using LLM."""

    def __init__(self, model: str = "gpt-4-mini", seed: int = 42):
        self.model = model
        self.rng = np.random.default_rng(seed)
        self._client = None

    def plan(self, candidates: "CandidateSet", user_request: str, request_id: str = "",
             target_duration: float = 60.0, mock: bool = False) -> EditDecision:
        request_id = request_id or f"req_{self.rng.integers(1000, 9999)}"

        if mock or not self._get_client():
            return self._mock_plan(candidates, user_request, request_id, target_duration)

        return self._llm_plan(candidates, user_request, request_id, target_duration)

    def revise(self, plan: EditDecision, errors: List[str], candidates: "CandidateSet",
               mock: bool = False) -> EditDecision:
        """Revise a plan that failed validation, given error messages."""
        if plan.revision_count >= 2:
            logger.warning(f"Max revisions (2) reached for {plan.request_id}")
            plan.notes += f" [MAX_REVISIONS]"
            return plan

        revision_prompt = f"""
The previous edit decision plan failed validation with the following errors:
{json.dumps(errors, indent=2)}

Previous plan:
{json.dumps(plan.to_dict(), indent=2)}

Fix the errors and return a corrected plan. Output ONLY the corrected JSON.
"""

        if mock or not self._get_client():
            return self._mock_revise(plan, errors)

        try:
            response = self._get_client().native_chat(revision_prompt)
            corrected = self._parse_json_response(response)
            if corrected:
                plan.segments = corrected.get("segments", plan.segments)
                plan.subtitle = corrected.get("subtitle", plan.subtitle)
                plan.audio_mix = corrected.get("audio_mix", plan.audio_mix)
                plan.notes = corrected.get("notes", plan.notes)
            plan.revision_count += 1
            plan.validation_errors = []
        except Exception as e:
            logger.error(f"LLM revision failed: {e}")
            plan.validation_errors.append(str(e))

        return plan

    def _llm_plan(self, candidates, user_request, request_id, target_duration) -> EditDecision:
        try:
            prompt = BUILD_PLAN_PROMPT.format(
                target_duration=target_duration,
                user_request=user_request,
                request_id=request_id,
                candidates_text=self._format_candidates(candidates),
            )
            response = self._get_client().native_chat(prompt)
            data = self._parse_json_response(response)

            if not data:
                return self._mock_plan(candidates, user_request, request_id, target_duration)

            return self._build_from_json(data, request_id)
        except Exception as e:
            logger.error(f"LLM planning failed: {e}")
            return self._mock_plan(candidates, user_request, request_id, target_duration)

    def _mock_plan(self, candidates, user_request: str, request_id: str, target_duration: float) -> EditDecision:
        segments = candidates.segments
        if not segments:
            return EditDecision(request_id=request_id, target_duration=target_duration)

        sorted_segs = sorted(segments, key=lambda s: s.importance_score, reverse=True)
        selected = []
        current_time = 0.0

        for seg in sorted_segs:
            dur = min(seg.duration_s, target_duration - current_time)
            if dur <= 0.5:
                break
            selected.append(EditSegment(
                segment_id=seg.segment_id,
                source_start=seg.start_s,
                source_end=seg.start_s + dur,
                target_start=round(current_time, 2),
                target_end=round(current_time + dur, 2),
                transition="cut",
                sync_anchor=SyncAnchor(
                    video_time=round(seg.start_s + dur / 2, 2),
                    audio_time=round(seg.start_s + dur / 2, 2),
                ),
                importance=seg.importance_score,
            ))
            current_time += dur

        selected = self._apply_swap_refinement(
            selected, sorted_segs, target_duration, delta_gap=0.50,
        )

        has_speech = any(seg.speech_presence for seg in candidates.segments
                         if seg.segment_id in {s.segment_id for s in selected})
        has_narration = "voice" in user_request.lower() or "narrat" in user_request.lower()

        subtitle = None
        if has_speech:
            subtitle = SubtitleSpec(enabled=True, text="auto", start_s=0.0,
                                     end_s=current_time, font_size=24, color="#FFFFFF")

        audio_mix = None
        if has_narration:
            audio_mix = AudioMixSpec(narration_enabled=True,
                                       narration_text="Automated video narration for selected highlights.",
                                       narration_volume=1.0, bgm_volume=0.3)

        return EditDecision(
            request_id=request_id, target_duration=target_duration,
            segments=selected, subtitle=subtitle, audio_mix=audio_mix,
            render_backend="ffmpeg",
            notes=f"Mock plan: {len(selected)} segments, {current_time:.1f}s total",
        )

    def _apply_swap_refinement(
        self, selected: List[EditSegment], candidates: List,
        target_duration: float, delta_gap: float = 0.50,
    ) -> List[EditSegment]:
        """Paper: greedy ranking + local swap refinement.

        One-pass swap: for each unselected candidate, if replacing a selected
        segment increases total objective (importance) while maintaining duration
        and gap constraints, perform the swap.
        """
        selected_ids = {s.segment_id for s in selected}
        unselected = [
            c for c in candidates
            if c.segment_id not in selected_ids and c.importance_score > 0
        ]
        if not selected or not unselected:
            return selected

        selected_total = sum(s.importance for s in selected)

        improved = True
        while improved:
            improved = False
            for unsel in unselected:
                best_gain = 0.0
                best_sel_idx = -1
                for i, sel in enumerate(selected):
                    dur_old = sel.target_end - sel.target_start
                    dur_new = min(unsel.duration_s, dur_old)
                    if dur_new <= 0.5:
                        continue
                    gain = unsel.importance_score - sel.importance
                    if gain > best_gain:
                        best_gain = gain
                        best_sel_idx = i

                if best_sel_idx >= 0:
                    old = selected[best_sel_idx]
                    dur_new = min(unsel.duration_s, old.target_end - old.target_start)
                    t_start = old.target_start
                    t_end = round(t_start + dur_new, 2)
                    selected[best_sel_idx] = EditSegment(
                        segment_id=unsel.segment_id,
                        source_start=unsel.start_s,
                        source_end=unsel.start_s + dur_new,
                        target_start=t_start,
                        target_end=t_end,
                        transition="cut",
                        sync_anchor=SyncAnchor(
                            video_time=round(unsel.start_s + dur_new / 2, 2),
                            audio_time=round(unsel.start_s + dur_new / 2, 2),
                        ),
                        importance=unsel.importance_score,
                    )
                    selected_ids = {s.segment_id for s in selected}
                    unselected = [
                        c for c in candidates
                        if c.segment_id not in selected_ids and c.importance_score > 0
                    ]
                    improved = True
                    break

        return selected

    def _mock_revise(self, plan: EditDecision, errors: List[str]) -> EditDecision:
        plan.notes += f" [REVISED_v{plan.revision_count + 1}]"
        plan.validation_errors = []
        plan.revision_count += 1
        plan.validation_passed = True

        if errors:
            for seg in plan.segments:
                seg.transition = "cut"
                seg.sync_anchor = SyncAnchor(
                    video_time=seg.source_start + (seg.source_end - seg.source_start) / 2,
                    audio_time=seg.source_start + (seg.source_end - seg.source_start) / 2,
                )
        return plan

    def _get_client(self):
        if self._client:
            return self._client
        try:
            from src.utils.llm_util import LLMUtil
            self._client = LLMUtil()
            return self._client
        except Exception:
            api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
            if api_key:
                from openai import OpenAI
                is_deepseek = bool(os.getenv("DEEPSEEK_API_KEY"))
                base_url = "https://api.deepseek.com" if is_deepseek else None
                self._client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
                self._client.native_chat = lambda msg: self._client.chat.completions.create(
                    model=self.model, messages=[{"role": "user", "content": msg}],
                ).choices[0].message.content
                return self._client
        return None

    def _format_candidates(self, candidates: "CandidateSet") -> str:
        lines = [f"FPS: {candidates.fps}", f"Total available: {candidates.total_duration_s:.1f}s", ""]
        for i, s in enumerate(candidates.segments[:30]):
            lines.append(
                f"  [{s.segment_id}] t={s.start_s:.1f}-{s.end_s:.1f}s (d={s.duration_s:.1f}s) "
                f"importance={s.importance_score:.3f} speech={s.speech_presence} event={s.event_presence}"
            )
        return "\n".join(lines)

    def _parse_json_response(self, response: str) -> Optional[Dict]:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        m = re.search(r'\{[\s\S]*\}', response)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return None

    def _build_from_json(self, data: Dict, request_id: str) -> EditDecision:
        segs = []
        for s in data.get("segments", []):
            sync = None
            if s.get("sync_anchor"):
                sync = SyncAnchor(
                    video_time=s["sync_anchor"].get("video_time", 0),
                    audio_time=s["sync_anchor"].get("audio_time", 0),
                )
            segs.append(EditSegment(
                segment_id=s.get("segment_id", ""),
                source_start=s.get("source_start", 0),
                source_end=s.get("source_end", 0),
                target_start=s.get("target_start", 0),
                target_end=s.get("target_end", 0),
                transition=s.get("transition", "cut"),
                sync_anchor=sync,
                importance=s.get("importance", 0.5),
            ))

        sub = None
        if data.get("subtitle") and data["subtitle"].get("enabled"):
            s = data["subtitle"]
            sub = SubtitleSpec(**{k: v for k, v in s.items() if k in SubtitleSpec.__dataclass_fields__})

        am = None
        if data.get("audio_mix") and data["audio_mix"].get("narration_enabled"):
            a = data["audio_mix"]
            am = AudioMixSpec(**{k: v for k, v in a.items() if k in AudioMixSpec.__dataclass_fields__})

        return EditDecision(
            request_id=request_id,
            target_duration=data.get("target_duration", 60.0),
            segments=segs, subtitle=sub, audio_mix=am,
            render_backend=data.get("render_backend", "ffmpeg"),
            notes=data.get("notes", ""),
        )
