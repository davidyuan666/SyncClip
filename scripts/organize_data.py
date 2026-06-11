#!/usr/bin/env python
"""
Organize video data files into genre-based directory structure per the paper.

Expected structure:
    experiments/data/
        vlog/            # 25 vlog videos

Usage:
    python scripts/organize_data.py                           # Reorganize current data
    python scripts/organize_data.py --data /path/to/videos     # From external dir
    python scripts/organize_data.py --check                    # Check structure only
    python scripts/organize_data.py --report                   # Report current contents
"""
import argparse
import json
import os
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GENRE_LIST = ["vlog"]
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".MP4", ".MOV"}
META_EXT = ".meta.json"

DEFAULT_DATA_DIR = "experiments/data"


def scan_videos(data_dir: Path) -> List[Path]:
    """Find all video files recursively, deduplicated by absolute path."""
    seen = set()
    paths = []
    for ext in VIDEO_EXTS:
        for p in data_dir.rglob(f"*{ext}"):
            abs_path = p.resolve()
            if abs_path not in seen:
                seen.add(abs_path)
                paths.append(p)
    return sorted(paths)


def classify_by_filename(path: Path) -> str:
    """Guess genre from filename or metadata."""
    name = path.stem.lower()
    keywords = {
        "vlog": ["vlog", "blog", "daily", "life", "talk", "tutorial"],
    }
    for genre, kws in keywords.items():
        for kw in kws:
            if kw in name:
                return genre
    return "vlog"


def try_load_meta_json(video_path: Path) -> dict:
    """Load associated .meta.json if it exists."""
    meta_path = video_path.with_suffix(META_EXT)
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    alt_meta = Path(str(video_path).replace(video_path.suffix, "") + META_EXT)
    if alt_meta != meta_path and alt_meta.exists():
        try:
            with open(alt_meta, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def organize(data_dir: Path, dry_run: bool = False) -> Dict[str, int]:
    """Move/copy videos into genre subdirectories."""
    if not dry_run:
        for genre in GENRE_LIST:
            (data_dir / genre).mkdir(parents=True, exist_ok=True)

    by_genre: Dict[str, int] = {g: 0 for g in GENRE_LIST}
    videos = scan_videos(data_dir)

    for vp in videos:
        genre = classify_by_filename(vp)
        dest_dir = data_dir / genre
        dest = dest_dir / vp.name

        if dry_run:
            print(f"  [{genre}] {vp.relative_to(data_dir)} -> {genre}/{vp.name}")
            by_genre[genre] += 1
            continue

        if vp.parent == dest_dir:
            by_genre[genre] += 1
            continue

        if dest.exists():
            dest = dest_dir / f"dup_{vp.name}"

        shutil.move(str(vp), str(dest))
        print(f"  MOVED: {vp.relative_to(data_dir)} -> {genre}/{vp.name}")

        meta_path = vp.with_suffix(META_EXT)
        if meta_path.exists():
            dest_meta = dest_dir / meta_path.name
            if dest_meta.exists():
                dest_meta = dest_dir / f"dup_{meta_path.name}"
            shutil.move(str(meta_path), str(dest_meta))

        by_genre[genre] += 1

    return by_genre


def check_structure(data_dir: Path) -> Dict[str, int]:
    """Verify genre directory structure and count videos."""
    counts: Dict[str, int] = {}
    for genre in GENRE_LIST:
        gdir = data_dir / genre
        if gdir.exists() and gdir.is_dir():
            n = len([f for f in gdir.iterdir() if f.suffix.lower() in VIDEO_EXTS])
            counts[genre] = n
        else:
            counts[genre] = 0
    return counts


def download_suggestions(data_dir: Path) -> None:
    """Suggest commands to fill missing genre videos."""
    from experiments.config import ExperimentConfig
    cfg = ExperimentConfig()
    target = cfg.genre_counts
    current = check_structure(data_dir)
    print("=" * 60)
    print("Video Genre Targets (from paper Table tab:dataset)")
    print("=" * 60)
    print(f"{'Genre':15s} {'Target':>7s} {'Current':>7s} {'Needed':>7s}")
    print("-" * 45)
    for genre in GENRE_LIST:
        t = target.get(genre, 10)
        c = current.get(genre, 0)
        n = max(0, t - c)
        flag = " <<<" if n > 0 else ""
        print(f"{genre:15s} {t:>7d} {c:>7d} {n:>7d}{flag}")

    total_needed = sum(max(0, target.get(g, 10) - current.get(g, 0)) for g in GENRE_LIST)
    if total_needed > 0:
        print(f"\n{total_needed} videos needed. Download with:")
        print(f"  python -m experiments.run_all --search \"<query>\" --source pexels --count <N> --download-only")


def main():
    parser = argparse.ArgumentParser(description="Organize SyncCLIPAgent video data by genre")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_DIR, help="Data directory")
    parser.add_argument("--dry-run", action="store_true", help="Preview without moving files")
    parser.add_argument("--check", action="store_true", help="Check structure only")
    parser.add_argument("--report", action="store_true", help="Report + download suggestions")
    args = parser.parse_args()

    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"ERROR: {data_dir} does not exist.")
        sys.exit(1)

    if args.report:
        check_structure(data_dir)
        download_suggestions(data_dir)
        return

    if args.check:
        print("=== Genre Structure Check ===")
        counts = check_structure(data_dir)
        for genre in GENRE_LIST:
            print(f"  {genre:15s}: {counts[genre]} videos")
        return

    print(f"Organizing videos in: {data_dir}")
    print("=" * 60)
    by_genre = organize(data_dir, dry_run=args.dry_run)
    print("=" * 60)

    total = sum(by_genre.values())
    for genre in GENRE_LIST:
        n = by_genre.get(genre, 0)
        print(f"  {genre:15s}: {n} videos")
    print(f"  {'TOTAL':15s}: {total} videos")

    from experiments.config import ExperimentConfig as _ExpCfg
    target = {g: c for g, c in zip(GENRE_LIST, _ExpCfg().genre_counts.values())}
    print("\nNote: Paper targets 100 videos total. Run with --report for download suggestions.")


if __name__ == "__main__":
    main()
