"""批量下载 100 个视频：混合模式 (IA + YouTube)"""
import os, sys, time, logging
from pathlib import Path

sys.path.insert(0, ".")

os.environ["HTTP_PROXY"] = "http://127.0.0.1:10809"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10809"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download")

from experiments.interactive import MIXED_GENRE_SOURCE_MAP, GENRE_LABELS, download_videos_by_genre
from experiments.config import ExperimentConfig


def _batch_progress_callback(i: int, total: int, meta) -> bool:
    title = meta["title"][:50] if meta else "???"
    genre = meta.get("genre", "?") if meta else "?"
    logger.info(f"  [{i + 1}/{total}] {title}  ({genre})")
    if (i + 1) % 10 == 0 and i + 1 < total:
        try:
            answer = input(f"\n>>> 已完成 {i + 1}/{total}，是否继续？[Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if answer in ("n", "no", "q", "quit"):
            return False
    return True


config = ExperimentConfig()
counts = dict(config.genre_counts)

total = sum(counts.values())
logger.info(f"开始下载 {total} 个视频 (混合模式)")

video_paths = download_videos_by_genre(
    counts, source="mixed", cache_dir=config.dataset_dir,
    progress_callback=_batch_progress_callback,
)

downloaded = sum(len(v) for v in video_paths.values())
logger.info(f"\n===== 下载完成: {downloaded}/{total} =====")
for genre in sorted(config.genre_list):
    paths = video_paths.get(genre, [])
    count = counts.get(genre, 0)
    label = GENRE_LABELS.get(genre, genre)
    src = MIXED_GENRE_SOURCE_MAP.get(genre, "?")
    ok = "OK" if len(paths) >= count else f"MISS {count - len(paths)}"
    logger.info(f"  {label:6s} ({src:7s}): {len(paths):>4d}/{count} {ok}")
