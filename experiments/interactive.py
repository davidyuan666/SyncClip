"""
SyncCLIPAgent 交互式实验启动脚本

特性:
  - API Key 仅存内存，不会写入文件
  - 交互式选择视频源、数量、运行模式
  - 支持 mock/real 两种模式

用法:
  python -m experiments.interactive
"""
from __future__ import annotations

import argparse
import getpass
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
logger = logging.getLogger("interactive")

DIVIDER = "=" * 56
SUB_DIVIDER = "-" * 56

GENRE_LABELS = {
    "vlog": "Vlog",
}

SOURCE_OPTIONS = {
    "1": ("mixed", "混合模式 (推荐: YouTube+Internet Archive 按类别自动分流)"),
    "2": ("archive", "Internet Archive (公版/CC0, 免API, 学术安全) [已验证]"),
    "3": ("youtube", "YouTube (yt-dlp, 需代理, 资源最丰富) [已验证]"),
    "4": ("bilibili", "Bilibili/B站 (当前不可用: API 412认证限制)"),
    "5": ("pexels", "Pexels (需免费注册API Key)"),
    "6": ("pixabay", "Pixabay (需API Key)"),
    "7": ("skip", "跳过下载，使用已有视频"),
}

MIXED_GENRE_SOURCE_MAP: Dict[str, str] = {
    "vlog": "youtube",
}

MODE_OPTIONS = {
    "1": ("mock", "Mock 模式 (无需GPU/API, 快速验证流水线)"),
    "2": ("real", "真实模式 (需GPU + DeepSeek API)"),
}


def _clear_screen():
    os.system("cls" if sys.platform == "win32" else "clear")


def _mask_key(key: str) -> str:
    if not key or len(key) < 8:
        return "***"
    return key[:3] + "****" + key[-4:]


def _read_int(prompt: str, default: int, min_val: int = 0, max_val: int = 1000) -> int:
    while True:
        raw = input(f"  {prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            val = int(raw)
            if min_val <= val <= max_val:
                return val
            print(f"    请输入 {min_val}-{max_val} 之间的整数")
        except ValueError:
            print(f"    请输入有效整数")


def _read_yes_no(prompt: str, default_yes: bool = True) -> bool:
    suffix = "[Y/n]" if default_yes else "[y/N]"
    raw = input(f"  {prompt} {suffix}: ").strip().lower()
    if not raw:
        return default_yes
    return raw in ("y", "yes")


def step_api_key() -> Optional[str]:
    print(f"\n{DIVIDER}")
    print("  [1/4] API 密钥")
    print(DIVIDER)
    print("  DeepSeek API Key 仅保存在内存中，不会写入任何文件。")
    print("  进程退出后即消失。\n")

    key = getpass.getpass("  DeepSeek API Key (sk-...): ").strip()
    if not key:
        print("  ! 未输入 API Key，将使用 Mock 模式运行 LLM 规划")
        return None

    os.environ["DEEPSEEK_API_KEY"] = key
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = key

    print(f"  OK 已设置: {_mask_key(key)}")
    return key


def step_video_source() -> Tuple[str, Optional[str]]:
    print(f"\n{DIVIDER}")
    print("  [2/4] 视频源选择")
    print(DIVIDER)
    for key, (name, desc) in SOURCE_OPTIONS.items():
        print(f"  [{key}] {desc}")

    while True:
        choice = input("\n  请选择 [1]: ").strip() or "1"
        if choice in SOURCE_OPTIONS:
            break
        print(f"    请输入 1-{len(SOURCE_OPTIONS)}")

    source, _ = SOURCE_OPTIONS[choice]

    extra_key = None
    if source == "pixabay":
        raw = getpass.getpass("  Pixabay API Key: ").strip()
        if raw:
            extra_key = raw
            os.environ["PIXABAY_API_KEY"] = raw
    elif source == "pexels":
        raw = getpass.getpass("  Pexels API Key (留空跳过): ").strip()
        if raw:
            extra_key = raw
            os.environ["PEXELS_API_KEY"] = raw

    return source, extra_key


def step_video_counts(current_counts: Dict[str, int]) -> Dict[str, int]:
    print(f"\n{DIVIDER}")
    print("  [3/4] 视频数量配置")
    print(DIVIDER)

    total_default = sum(current_counts.values())
    print(f"  论文标准: 总计 {total_default} 个视频\n")
    print(f"  {'类别':<12s} {'数量':>6s}  {'中文':>8s}")
    print(f"  {SUB_DIVIDER}")
    for genre in sorted(current_counts.keys()):
        print(f"  {genre:<12s} {current_counts[genre]:>5d}   {GENRE_LABELS.get(genre, genre):>8s}")
    print(f"  {SUB_DIVIDER}")
    print(f"  {'总计':<12s} {total_default:>5d}")

    if _read_yes_no("使用默认数量?", default_yes=True):
        return current_counts

    choice = input("\n  自定义方式: [1]全局统一数量 [2]逐类设置 [1]: ").strip() or "1"

    if choice == "1":
        n = _read_int("每类视频数量", default=10, min_val=1, max_val=100)
        return {g: n for g in current_counts}

    new_counts = {}
    for genre in sorted(current_counts.keys()):
        label = GENRE_LABELS.get(genre, genre)
        new_counts[genre] = _read_int(f"{genre} ({label})", default=current_counts[genre], min_val=0, max_val=100)
    return new_counts


def step_run_mode() -> str:
    print(f"\n{DIVIDER}")
    print("  [4/4] 运行模式")
    print(DIVIDER)
    for key, (mode, desc) in MODE_OPTIONS.items():
        print(f"  [{key}] {desc}")

    while True:
        choice = input("\n  请选择 [1]: ").strip() or "1"
        if choice in MODE_OPTIONS:
            break
        print("    请输入 1 或 2")

    mode, _ = MODE_OPTIONS[choice]
    return mode


def print_summary(
    api_key: Optional[str],
    source: str,
    counts: Dict[str, int],
    mode: str,
    output_dir: str,
) -> None:
    print(f"\n{DIVIDER}")
    print("  配置确认")
    print(DIVIDER)

    has_key = "已设置" if api_key else "未设置 (将使用Mock LLM)"
    key_display = _mask_key(api_key) if api_key else "N/A"

    source_label = {"pexels": "Pexels", "pixabay": "Pixabay", "skip": "跳过下载",
                    "bilibili": "Bilibili/B站", "youtube": "YouTube", "archive": "Internet Archive",
                    "mixed": "混合模式 (IA+YouTube)"}.get(source, source)
    mode_label = {"mock": "Mock 模式 (快速验证)", "real": "真实模式"}.get(mode, mode)
    total = sum(counts.values())

    print(f"  API Key:      {has_key} ({key_display})")
    print(f"  视频源:       {source_label}")
    print(f"  总视频数:     {total}")
    print(f"  各类型数量:   {dict((GENRE_LABELS.get(g, g), c) for g, c in counts.items() if c > 0)}")
    if source == "mixed":
        ia_genres = [GENRE_LABELS.get(g, g) for g in counts if MIXED_GENRE_SOURCE_MAP.get(g) == "archive" and counts[g] > 0]
        yt_genres = [GENRE_LABELS.get(g, g) for g in counts if MIXED_GENRE_SOURCE_MAP.get(g) == "youtube" and counts[g] > 0]
        print(f"    IA 负责:     {', '.join(ia_genres) if ia_genres else '(无)'}")
        print(f"    YT 负责:     {', '.join(yt_genres) if yt_genres else '(无)'}")
    print(f"  运行模式:     {mode_label}")
    print(f"  输出目录:     {output_dir}")
    print(DIVIDER)


def download_videos_by_genre(
    counts: Dict[str, int],
    source: str = "mixed",
    cache_dir: str = "experiments/data",
    progress_callback = None,
) -> Dict[str, List[str]]:
    from experiments.video_downloader import download_videos, GENRE_KEYWORDS, IA_GENRE_KEYWORDS, YtDlpDownloader

    video_paths: Dict[str, List[str]] = {}
    total = sum(counts.values())

    for genre, count in counts.items():
        if count <= 0:
            continue

        if source == "mixed":
            actual_source = MIXED_GENRE_SOURCE_MAP.get(genre, "youtube")
        else:
            actual_source = source

        if actual_source == "archive":
            keywords_map = IA_GENRE_KEYWORDS
        elif actual_source == "bilibili":
            keywords_map = YtDlpDownloader.BILIBILI_GENRE_KEYWORDS
        else:
            keywords_map = GENRE_KEYWORDS

        keywords = keywords_map.get(genre, [genre])
        query = " ".join(keywords[:3])
        logger.info(f"[{genre}] 搜索: '{query}' x {count} (source={actual_source})")

        try:
            paths, metas = download_videos(
                query=query, source=actual_source, count=count, cache_dir=cache_dir,
                progress_callback=progress_callback,
            )
            if paths:
                for g, plist in paths.items():
                    video_paths.setdefault(g, []).extend(plist)
            logger.info(f"  [{genre}] 下载完成: {len(metas)} 个视频, "
                         f"归类为: {dict((g, len(p)) for g, p in paths.items())}")
        except Exception as e:
            logger.error(f"  [{genre}] 下载失败: {e}")

        if count > 1:
            time.sleep(1.5)

    downloaded = sum(len(v) for v in video_paths.values())
    logger.info(f"总计下载: {downloaded}/{total} 个视频")
    return video_paths


def run_experiments(
    config,
    video_paths: Dict[str, List[str]],
) -> None:
    from experiments.runner import ExperimentRunner
    from experiments.ground_truth import GroundTruthBuilder
    from experiments.sensitivity import SensitivityAnalyzer
    from experiments.profiling import RuntimeProfiler
    from experiments.robustness import RobustnessTester
    from experiments.visualization import ExperimentVisualizer

    os.makedirs(config.output_dir, exist_ok=True)
    config.save(os.path.join(config.output_dir, "config.json"))

    logger.info("=" * 56)
    logger.info("开始实验流水线")
    logger.info("=" * 56)

    from experiments.run_all import _generate_mock_segments
    mock_segments = _generate_mock_segments(config)

    gt_builder = GroundTruthBuilder(config)
    gt_result, ground_truth = gt_builder.build_ground_truth(mock_segments, n_real=40, n_synthetic=60)
    logger.info(f"Ground Truth: kappa={gt_result.cohens_kappa:.4f}, segments={gt_result.n_total_segments}")

    runner = ExperimentRunner(config)
    pipeline_result = runner.run_full_pipeline(
        video_paths=video_paths,
        reference_segments=ground_truth,
    )
    logger.info(f"Pipeline: P={pipeline_result.segment_metrics.precision:.4f}, "
                 f"R={pipeline_result.segment_metrics.recall:.4f}, "
                 f"F1={pipeline_result.segment_metrics.f1_score:.4f}")

    logger.info("--- 敏感性分析 ---")
    sensitivity = SensitivityAnalyzer(config)
    sensitivity.run_all()

    logger.info("--- 运行时分析 ---")
    profiler = RuntimeProfiler(config)
    profiler.profile_all()

    logger.info("--- 鲁棒性测试 ---")
    robustness = RobustnessTester(config)
    robustness.evaluate()

    logger.info("--- 可视化 ---")
    viz = ExperimentVisualizer(str(Path(config.output_dir)))
    viz.save_all()


def run(args: argparse.Namespace) -> None:
    _clear_screen()
    print(DIVIDER)
    print("  SyncCLIPAgent 实验交互式启动")
    print(DIVIDER)

    from experiments.config import load_config

    api_key = step_api_key()

    source, _ = step_video_source()

    config = load_config()
    counts = step_video_counts(dict(config.genre_counts))

    mode = step_run_mode()
    is_mock = mode == "mock"

    config.mock_mode = is_mock
    config.output_dir = args.output
    if api_key:
        config.openai_api_key = api_key
        config.llm_enabled = True

    print_summary(api_key, source, counts, mode, config.output_dir)

    if not _read_yes_no("\n确认开始运行?", default_yes=True):
        print("\n已取消。")
        return

    start_time = datetime.now()
    logger.info(f"开始时间: {start_time}")

    video_paths: Dict[str, List[str]] = {}

    if source != "skip":
        logger.info("--- 下载视频 ---")
        download_root = config.dataset_dir if source == "mixed" else os.path.join(config.dataset_dir, source)
        video_paths = download_videos_by_genre(counts, source=source, cache_dir=download_root)
        if source == "mixed":
            ia_count = sum(c for g, c in counts.items() if MIXED_GENRE_SOURCE_MAP.get(g) == "archive")
            yt_count = sum(counts.values()) - ia_count
            logger.info(f"混合分流: Internet Archive={ia_count}, YouTube={yt_count}")
    else:
        for genre in config.genre_list:
            genre_dir = Path(config.dataset_dir) / source or Path(config.dataset_dir)
            search_dir = genre_dir / genre
            if search_dir.exists():
                mp4s = list(search_dir.glob("*.mp4"))
                if mp4s:
                    video_paths[genre] = [str(p) for p in mp4s]
        if not video_paths:
            root = Path(config.dataset_dir)
            for p in root.rglob("*.mp4"):
                video_paths.setdefault("vlog", []).append(str(p))
        logger.info(f"使用已有视频: {sum(len(v) for v in video_paths.values())} 个")

    total_videos = sum(len(v) for v in video_paths.values())
    if total_videos == 0:
        logger.warning("没有视频可用，将使用 Mock 数据进行实验")

    run_experiments(config, video_paths)

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"\n所有实验完成! 耗时: {elapsed:.1f}s")
    logger.info(f"结果保存在: {config.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="SyncCLIPAgent 交互式实验启动器")
    parser.add_argument("--output", type=str, default="experiments/output",
                        help="输出目录 (默认: experiments/output)")
    parser.add_argument("--no-interactive", action="store_true",
                        help="非交互模式 (使用 --mock 和默认值)")
    parser.add_argument("--mock", action="store_true", default=True,
                        help="Mock 模式 (默认)")
    parser.add_argument("--api-key", type=str, default="",
                        help="DeepSeek API Key (非交互模式)")
    args = parser.parse_args()

    if args.no_interactive:
        from experiments.config import load_config
        config = load_config(mock_mode=args.mock, output_dir=args.output)
        if args.api_key:
            os.environ["DEEPSEEK_API_KEY"] = args.api_key
            os.environ["OPENAI_API_KEY"] = args.api_key
            config.openai_api_key = args.api_key
        logger.info(f"非交互模式: mock={args.mock}")
        run_experiments(config, {})
        return

    run(args)


if __name__ == "__main__":
    main()
