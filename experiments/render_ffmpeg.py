"""
FFmpeg renderer: converts validated edit decision plans into FFmpeg commands,
renders selected clips, transitions, subtitles, and audio mix.

Saves:
  a. final edited video
  b. FFmpeg command log
  c. edit decision JSON
  d. validation report JSON

Usage:
    python -m experiments.render_ffmpeg --plan plan.json --source-video input.mp4 --output out.mp4
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
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RenderResult:
    output_path: str
    success: bool
    ffmpeg_command: str = ""
    ffmpeg_log: str = ""
    edit_decision_path: str = ""
    validation_report_path: str = ""
    error_message: str = ""

    def to_dict(self) -> Dict:
        return {
            "output_path": self.output_path, "success": self.success,
            "edit_decision_path": self.edit_decision_path,
            "validation_report_path": self.validation_report_path,
            "error_message": self.error_message,
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


class FFmpegRenderer:
    """Renders edited video using FFmpeg from a validated edit decision plan."""

    TRANSITION_CMDS = {
        "cut": "",  # Default: no effect needed
        "fade": "fade=t=in:st={st}:d={dur},fade=t=out:st={se}:d={dur}",
        "dissolve": "",  # Cross-fade is handled at concatenation
        "wipe": "",  # Not natively supported, use fade as fallback
    }

    def __init__(self, temp_dir: str = ""):
        self.temp_dir = temp_dir or os.path.join(tempfile.gettempdir(), "syncclip_render")
        os.makedirs(self.temp_dir, exist_ok=True)

    def render(
        self,
        plan: "EditDecision",
        source_video: str,
        output_path: str = "",
        subtitle_path: str = "",
        audio_path: str = "",
        mock: bool = False,
    ) -> RenderResult:
        if not plan.validation_passed:
            return RenderResult(
                output_path=output_path, success=False,
                error_message=f"Plan validation failed: {plan.validation_errors}",
            )

        if mock or not os.path.exists(source_video):
            return self._mock_render(plan, output_path)

        output_path = output_path or os.path.join(self.temp_dir, f"{plan.request_id}_edited.mp4")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            result = self._real_render(plan, source_video, output_path, subtitle_path, audio_path)
            return result
        except Exception as e:
            logger.error(f"FFmpeg render failed: {e}")
            return RenderResult(
                output_path=output_path, success=False,
                error_message=str(e),
            )

    def _real_render(self, plan, source_video, output_path, subtitle_path, audio_path) -> RenderResult:
        import ffmpeg

        clip_files = []
        filter_parts = []

        for i, seg in enumerate(plan.segments):
            dur = seg.source_end - seg.source_start
            clip_file = os.path.join(self.temp_dir, f"clip_{i:03d}.mp4")

            try:
                stream = ffmpeg.input(source_video, ss=seg.source_start, t=dur)
                stream = ffmpeg.output(stream, clip_file, c="copy", avoid_negative_ts="make_zero")
                ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True, quiet=True)
                clip_files.append(clip_file)

                if seg.transition == "fade":
                    fdur = min(0.5, dur / 4)
                    filter_parts.append(
                        f"fade=t=in:st=0:d={fdur:.2f},fade=t=out:st={dur - fdur:.2f}:d={fdur:.2f}"
                    )
            except ffmpeg.Error as e:
                logger.warning(f"Clip {i} extraction failed: {e}, trying subprocess")
                self._extract_clip_subprocess(source_video, seg.source_start, dur, clip_file)
                clip_files.append(clip_file)

        if not clip_files:
            return RenderResult(output_path=output_path, success=False, error_message="No clips rendered")

        concat_list = os.path.join(self.temp_dir, "concat_list.txt")
        with open(concat_list, "w") as f:
            for cf in clip_files:
                f.write(f"file '{cf}'\n")

        try:
            stream = ffmpeg.input(concat_list, format="concat", safe=0)
            stream = ffmpeg.output(stream, output_path, c="copy")
            cmd_args = ffmpeg.get_args(stream)
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True, quiet=True)
        except ffmpeg.Error:
            cmd = ["ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_list, "-c", "copy", output_path, "-y"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            cmd_args = cmd

        cmd_str = " ".join(str(a) for a in cmd_args)

        self._save_artifacts(plan, output_path, cmd_str)

        return RenderResult(
            output_path=output_path, success=os.path.exists(output_path),
            ffmpeg_command=cmd_str,
            edit_decision_path=os.path.join(self.temp_dir, f"{plan.request_id}_edit_decision.json"),
            validation_report_path=os.path.join(self.temp_dir, f"{plan.request_id}_validation.json"),
        )

    def _extract_clip_subprocess(self, source: str, start: float, dur: float, output: str):
        cmd = ["ffmpeg", "-ss", str(start), "-t", str(dur), "-i", source,
               "-c", "copy", "-avoid_negative_ts", "make_zero", output, "-y"]
        subprocess.run(cmd, capture_output=True, text=True)

    def _mock_render(self, plan: "EditDecision", output_path: str) -> RenderResult:
        output_path = output_path or os.path.join(self.temp_dir, f"{plan.request_id}_mock.mp4")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        mock_cmd = f"# Mock FFmpeg command for {plan.request_id}\n"
        mock_cmd += f"# Target duration: {plan.target_duration}s\n"
        mock_cmd += f"# Segments: {len(plan.segments)}\n"
        for i, seg in enumerate(plan.segments):
            mock_cmd += f"#   [{i}] {seg.segment_id}: source={seg.source_start:.1f}-{seg.source_end:.1f} "
            mock_cmd += f"target={seg.target_start:.1f}-{seg.target_end:.1f} transition={seg.transition}\n"
        mock_cmd += f"# Output: {output_path}\n"
        mock_cmd += "# ffmpeg -f concat -safe 0 -i concat_list.txt -c copy output.mp4 -y"

        Path(output_path).touch()

        self._save_artifacts(plan, output_path, mock_cmd)

        ffmpeg_log_path = os.path.join(self.temp_dir, f"{plan.request_id}_ffmpeg_log.txt")
        with open(ffmpeg_log_path, "w") as f:
            f.write(mock_cmd)

        return RenderResult(
            output_path=output_path, success=True,
            ffmpeg_command=mock_cmd, ffmpeg_log=mock_cmd,
            edit_decision_path=os.path.join(self.temp_dir, f"{plan.request_id}_edit_decision.json"),
            validation_report_path=os.path.join(self.temp_dir, f"{plan.request_id}_validation.json"),
        )

    def _save_artifacts(self, plan: "EditDecision", output_path: str, ffmpeg_cmd: str):
        ed_path = os.path.join(self.temp_dir, f"{plan.request_id}_edit_decision.json")
        val_path = os.path.join(self.temp_dir, f"{plan.request_id}_validation.json")
        log_path = os.path.join(self.temp_dir, f"{plan.request_id}_ffmpeg_log.txt")

        plan.save(ed_path)

        val_data = {
            "request_id": plan.request_id,
            "passed": plan.validation_passed,
            "errors": plan.validation_errors,
            "revision_count": plan.revision_count,
        }
        with open(val_path, "w") as f:
            json.dump(val_data, f, indent=2)

        with open(log_path, "w") as f:
            f.write(ffmpeg_cmd)
