"""
Online video downloader: supports Pexels, Pixabay, Internet Archive,
and generic yt-dlp for YouTube/Bilibili/Vimeo.

Integrates directly with the SyncCLIPAgent experiment pipeline:
    download(url) -> local path -> ExperimentRunner.run_full_pipeline(...)

Usage:
    python -m experiments.run_all --url "https://pexels.com/video/..." --mock
    python -m experiments.run_all --search "action sports" --source pexels --count 5
    python -m experiments.run_all --search "documentary" --source pixabay --count 3 --download-only
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import requests

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "data"))

GENRE_KEYWORDS: Dict[str, List[str]] = {
    "action": ["action", "fight", "explosion", "chase", "racing", "sport", "parkour", "stunt", "battle"],
    "documentary": ["documentary", "nature", "wildlife", "history", "science", "education", "interview"],
    "vlog": ["vlog", "daily", "routine", "lifestyle", "tutorial", "review", "unboxing", "blog"],
    "news": ["news", "report", "headline", "breaking", "journalist", "anchor", "broadcast", "press"],
    "sports": ["sports", "football", "basketball", "soccer", "tennis", "swimming", "game", "match", "athlete"],
    "music_video": ["music", "song", "dance", "band", "concert", "performance", "lyric", "mv", "official video"],
    "short_film": ["short film", "cinematic", "story", "drama", "narrative", "fiction", "art film", "experimental"],
}

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)


def _url_hash(url: str) -> str:
    return hashlib.md5(url.encode("utf-8")).hexdigest()[:12]


def _sanitize_filename(name: str, max_len: int = 80) -> str:
    name = re.sub(r'[<>:"/\\|?*\u0000-\u001f\u007f-\u009f]', "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    if name:
        name = name.encode("ascii", errors="replace").decode("ascii", errors="replace")
        name = name.replace("?", "_")
    return name[:max_len] if name else "video"


def _classify_genre(title: str, tags: Optional[List[str]] = None) -> str:
    text = (title + " " + " ".join(tags or [])).lower()
    scores = {}
    for genre, keywords in GENRE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        scores[genre] = score
    if max(scores.values()) == 0:
        return "vlog"
    return max(scores, key=scores.get)


def _check_ytdlp() -> bool:
    try:
        subprocess.run([sys.executable, "-m", "yt_dlp", "--version"],
                       capture_output=True, text=True, timeout=10)
        return True
    except Exception:
        return False


class VideoSource(ABC):
    """Abstract base for video download sources."""

    def __init__(self, cache_dir: str = "", source_name: str = "generic"):
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
        self.source_name = source_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def supports(self, url: str) -> bool:
        ...

    @abstractmethod
    def download(self, url: str, output_dir: Optional[str] = None) -> Optional[Dict]:
        """
        Download video and return metadata dict:
            {path, url, title, duration, genre, source, thumbnail}
        Returns None on failure.
        """
        ...

    @abstractmethod
    def search(self, query: str, count: int = 5, **kwargs) -> List[str]:
        """Search and return list of video URLs."""
        ...

    def _get_cache_path(self, url: str) -> Path:
        return self.cache_dir / self.source_name / f"{_url_hash(url)}.mp4"

    def _is_cached(self, url: str) -> Optional[Path]:
        cache_path = self._get_cache_path(url)
        meta_path = cache_path.with_suffix(".meta.json")
        if cache_path.exists() and cache_path.stat().st_size > 1024:
            return cache_path
        return None

    def _save_meta(self, cache_path: Path, meta: Dict) -> None:
        meta_path = cache_path.with_suffix(".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)


class YtDlpDownloader(VideoSource):
    """Generic downloader using yt-dlp. Supports YouTube, Bilibili, Vimeo, etc."""

    YTDLP_DOMAINS = [
        "youtube.com", "youtu.be", "bilibili.com", "vimeo.com",
        "dailymotion.com", "twitch.tv", "nicovideo.jp",
    ]

    def __init__(self, cache_dir: str = ""):
        super().__init__(cache_dir, source_name="ytdlp")

    def supports(self, url: str) -> bool:
        if not _check_ytdlp():
            return False
        return any(d in url.lower() for d in self.YTDLP_DOMAINS)

    def download(self, url: str, output_dir: Optional[str] = None) -> Optional[Dict]:
        if not _check_ytdlp():
            logger.error("yt-dlp not installed. Run: pip install yt-dlp")
            return None

        cache_path = self._get_cache_path(url)
        cached = self._is_cached(url)
        if cached:
            logger.info(f"[ytdlp] Using cached: {cached}")
            return self._load_cached_meta(cached, url)

        output_dir = Path(output_dir or (self.cache_dir / self.source_name))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_template = str(output_dir / f"{_url_hash(url)}.%(ext)s")

        try:
            cmd = [
                sys.executable, "-m", "yt_dlp",
                "--format", "best[height<=720][ext=mp4]/best[height<=720]/best",
                "--output", output_template,
                "--merge-output-format", "mp4",
                "--no-playlist",
                "--socket-timeout", "30",
                "--retries", "3",
                "--print", "title",
                "--print", "duration",
                "--print", "thumbnail",
                url,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                logger.error(f"[ytdlp] Download failed: {result.stderr[:500]}")
                return None

            lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
            title = lines[0] if lines else url
            duration = float(lines[1]) if len(lines) > 1 else 0.0
            thumbnail = lines[2] if len(lines) > 2 else ""

            actual_path = None
            for ext in [".mp4", ".mkv", ".webm"]:
                candidate = output_dir / f"{_url_hash(url)}{ext}"
                if candidate.exists():
                    actual_path = candidate
                    break
            if not actual_path:
                matches = list(output_dir.glob(f"{_url_hash(url)}*"))
                actual_path = matches[0] if matches else Path(output_template.replace("%(ext)s", "mp4"))

            if not actual_path or not actual_path.exists():
                return None

            if not actual_path.suffix == ".mp4":
                new_path = actual_path.with_suffix(".mp4")
                shutil.move(str(actual_path), str(new_path))
                actual_path = new_path

            genre = _classify_genre(title)
            meta = {
                "path": str(actual_path), "url": url, "title": title,
                "duration": duration, "genre": genre, "source": "ytdlp",
                "thumbnail": thumbnail,
            }
            self._save_meta(actual_path, meta)
            logger.info(f"[ytdlp] Downloaded: {title[:60]} -> {actual_path}")
            return meta

        except subprocess.TimeoutExpired:
            logger.error(f"[ytdlp] Timeout for: {url}")
            return None
        except Exception as e:
            logger.error(f"[ytdlp] Error: {e}")
            return None

    def search(self, query: str, count: int = 5, **kwargs) -> List[str]:
        if not _check_ytdlp():
            return []

        try:
            cmd = [
                sys.executable, "-m", "yt_dlp",
                f"ytsearch{count}:{query}",
                "--get-url",
                "--no-playlist",
                "--flat-playlist",
                "--socket-timeout", "15",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                return []
            urls = [l.strip() for l in result.stdout.strip().split("\n") if l.strip().startswith("http")]
            return urls[:count]
        except Exception as e:
            logger.error(f"[ytdlp] Search error: {e}")
            return []

    def _load_cached_meta(self, cache_path: Path, url: str) -> Optional[Dict]:
        meta_path = cache_path.with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "path": str(cache_path), "url": url, "title": cache_path.stem,
            "duration": 0, "genre": "vlog", "source": "ytdlp",
        }


class PexelsDownloader(VideoSource):
    """Pexels free stock video downloader. API is open (rate-limited without key)."""

    API_BASE = "https://api.pexels.com/videos"
    PEXELS_DOMAINS = ["pexels.com", "www.pexels.com"]

    def __init__(self, api_key: str = "", cache_dir: str = ""):
        super().__init__(cache_dir, source_name="pexels")
        self.api_key = api_key or os.getenv("PEXELS_API_KEY", "")
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": _USER_AGENT})
        if self.api_key:
            self._session.headers["Authorization"] = self.api_key

    def supports(self, url: str) -> bool:
        return any(d in url.lower() for d in self.PEXELS_DOMAINS)

    def download(self, url: str, output_dir: Optional[str] = None) -> Optional[Dict]:
        cached = self._is_cached(url)
        if cached:
            logger.info(f"[pexels] Using cached: {cached}")
            return self._load_cached_meta(cached, url)

        video_id = self._extract_id(url)
        if not video_id:
            video_id = url

        try:
            api_url = f"{self.API_BASE}/videos/{video_id}" if isinstance(video_id, int) else self.API_BASE + "/search"
            if isinstance(video_id, int):
                resp = self._session.get(api_url, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                video_files = sorted(
                    data.get("video_files", []),
                    key=lambda f: (f.get("height", 0) * (-1), f.get("width", 0) * (-1)),
                )
                hd_file = next((f for f in video_files if f.get("height", 0) <= 720), video_files[0] if video_files else None)
                if not hd_file:
                    return None
                download_url = hd_file["link"]
                title = _sanitize_filename(data.get("user", {}).get("name", "pexels")) + "_" + str(data.get("id", ""))
                duration = data.get("duration", 0)
                tags = data.get("tags", [])
            else:
                resp = self._session.get(api_url, params={"query": video_id, "per_page": 1}, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                videos = data.get("videos", [])
                if not videos:
                    return None
                v = videos[0]
                video_files = sorted(
                    v.get("video_files", []),
                    key=lambda f: (f.get("height", 0) * (-1),),
                )
                hd_file = next((f for f in video_files if f.get("height", 0) <= 720), video_files[0] if video_files else None)
                if not hd_file:
                    return None
                download_url = hd_file["link"]
                title = _sanitize_filename(v.get("user", {}).get("name", "pexels")) + "_" + str(v.get("id", ""))
                duration = v.get("duration", 0)
                tags = [t.lower() for t in v.get("tags", [])] if v.get("tags") else []

            cache_path = self._get_cache_path(url)
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            dl_resp = self._session.get(download_url, timeout=120, stream=True)
            dl_resp.raise_for_status()
            with open(cache_path, "wb") as f:
                for chunk in dl_resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            genre = _classify_genre(title, tags)
            meta = {
                "path": str(cache_path), "url": url, "title": title,
                "duration": duration, "genre": genre, "source": "pexels",
                "tags": tags,
            }
            self._save_meta(cache_path, meta)
            logger.info(f"[pexels] Downloaded: {title[:60]} -> {cache_path}")
            return meta

        except requests.RequestException as e:
            logger.warning(f"[pexels] API error: {e}, trying yt-dlp fallback")
            return self._fallback_ytdlp(url)
        except Exception as e:
            logger.error(f"[pexels] Error: {e}")
            return None

    def search(self, query: str, count: int = 5, **kwargs) -> List[str]:
        try:
            if self.api_key:
                resp = self._session.get(
                    f"{self.API_BASE}/search",
                    params={"query": query, "per_page": min(count, 80)},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                videos = data.get("videos", [])
                urls = [f"https://www.pexels.com/video/{v['id']}" for v in videos]
            else:
                resp = self._session.get(
                    f"{self.API_BASE}/search",
                    params={"query": query, "per_page": min(count, 80)},
                    headers={k: v for k, v in self._session.headers.items() if k != "Authorization"},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                videos = data.get("videos", [])
                urls = [f"https://www.pexels.com/video/{v['id']}" for v in videos]

            if not urls and _check_ytdlp():
                fallback = YtDlpDownloader(str(self.cache_dir.parent))
                urls = fallback.search(f"{query} site:pexels.com", count)

            return urls[:count]
        except Exception as e:
            logger.warning(f"[pexels] Search error: {e}")
            return []

    def _extract_id(self, url: str) -> Optional[int]:
        m = re.search(r'pexels\.com/video/(\d+)', url)
        return int(m.group(1)) if m else None

    def _fallback_ytdlp(self, url: str) -> Optional[Dict]:
        if _check_ytdlp():
            fallback = YtDlpDownloader(str(self.cache_dir.parent))
            return fallback.download(url)
        return None

    def _load_cached_meta(self, cache_path: Path, url: str) -> Optional[Dict]:
        meta_path = cache_path.with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"path": str(cache_path), "url": url, "title": cache_path.stem,
                "duration": 0, "genre": "vlog", "source": "pexels"}


class PixabayDownloader(VideoSource):
    """Pixabay free stock video downloader."""

    API_BASE = "https://pixabay.com/api/videos/"
    PIXABAY_DOMAINS = ["pixabay.com", "www.pixabay.com"]

    def __init__(self, api_key: str = "", cache_dir: str = ""):
        super().__init__(cache_dir, source_name="pixabay")
        self.api_key = api_key or os.getenv("PIXABAY_API_KEY", "")
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": _USER_AGENT})

    def supports(self, url: str) -> bool:
        return any(d in url.lower() for d in self.PIXABAY_DOMAINS)

    def download(self, url: str, output_dir: Optional[str] = None) -> Optional[Dict]:
        cached = self._is_cached(url)
        if cached:
            logger.info(f"[pixabay] Using cached: {cached}")
            return self._load_cached_meta(cached, url)

        video_id = self._extract_id(url)

        if video_id and self.api_key:
            resp = self._session.get(
                self.API_BASE,
                params={"key": self.api_key, "id": video_id},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            hits = data.get("hits", [])
            if not hits:
                return self._fallback_ytdlp(url)
            v = hits[0]
            videos = v.get("videos", {})
            hd = videos.get("large", {})
            if not hd:
                hd = videos.get("medium", {})
            download_url = hd.get("url", "")
            title = _sanitize_filename(v.get("tags", "pixabay"))
            tags = v.get("tags", "").split(",") if v.get("tags") else []
        elif self.api_key:
            resp = self._session.get(
                self.API_BASE,
                params={"key": self.api_key, "q": url, "per_page": 1},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            hits = data.get("hits", [])
            if not hits:
                return self._fallback_ytdlp(url)
            v = hits[0]
            videos = v.get("videos", {})
            hd = videos.get("large", {}) or videos.get("medium", {})
            download_url = hd.get("url", "")
            title = _sanitize_filename(v.get("tags", "pixabay"))
            tags = v.get("tags", "").split(",") if v.get("tags") else []
        else:
            return self._fallback_ytdlp(url)

        if not download_url:
            return self._fallback_ytdlp(url)
        if download_url.startswith("//"):
            download_url = "https:" + download_url

        cache_path = self._get_cache_path(url)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        dl_resp = self._session.get(download_url, timeout=120, stream=True)
        dl_resp.raise_for_status()
        with open(cache_path, "wb") as f:
            for chunk in dl_resp.iter_content(chunk_size=8192):
                f.write(chunk)

        genre = _classify_genre(title, tags)
        meta = {
            "path": str(cache_path), "url": url, "title": title,
            "duration": 0, "genre": genre, "source": "pixabay",
            "tags": tags,
        }
        self._save_meta(cache_path, meta)
        logger.info(f"[pixabay] Downloaded: {title[:60]} -> {cache_path}")
        return meta

    def search(self, query: str, count: int = 5, **kwargs) -> List[str]:
        if not self.api_key:
            if _check_ytdlp():
                fallback = YtDlpDownloader(str(self.cache_dir.parent))
                return fallback.search(f"{query} site:pixabay.com", count)
            return []

        try:
            resp = self._session.get(
                self.API_BASE,
                params={"key": self.api_key, "q": query, "per_page": min(count, 200), "video_type": "all"},
                timeout=30,
            )
            resp.raise_for_status()
            hits = resp.json().get("hits", [])
            urls = [f"https://pixabay.com/videos/{h['id']}" for h in hits]
            return urls[:count]
        except Exception as e:
            logger.warning(f"[pixabay] Search error: {e}")
            return []

    def _extract_id(self, url: str) -> Optional[str]:
        m = re.search(r'pixabay\.com/videos/(\d+)', url)
        return m.group(1) if m else None

    def _fallback_ytdlp(self, url: str) -> Optional[Dict]:
        if _check_ytdlp():
            fallback = YtDlpDownloader(str(self.cache_dir.parent))
            return fallback.download(url)
        return None

    def _load_cached_meta(self, cache_path: Path, url: str) -> Optional[Dict]:
        meta_path = cache_path.with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"path": str(cache_path), "url": url, "title": cache_path.stem,
                "duration": 0, "genre": "vlog", "source": "pixabay"}


class InternetArchiveDownloader(VideoSource):
    """Internet Archive video downloader."""

    IA_DOMAINS = ["archive.org", "www.archive.org"]

    def __init__(self, cache_dir: str = ""):
        super().__init__(cache_dir, source_name="archive")
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": _USER_AGENT})

    def supports(self, url: str) -> bool:
        return any(d in url.lower() for d in self.IA_DOMAINS)

    def download(self, url: str, output_dir: Optional[str] = None) -> Optional[Dict]:
        cached = self._is_cached(url)
        if cached:
            logger.info(f"[archive] Using cached: {cached}")
            return self._load_cached_meta(cached, url)

        identifier = self._extract_identifier(url)
        if not identifier:
            return None

        try:
            metadata_url = f"https://archive.org/metadata/{identifier}"
            resp = self._session.get(metadata_url, timeout=30)
            resp.raise_for_status()
            metadata = resp.json()

            title = _sanitize_filename(
                metadata.get("metadata", {}).get("title", identifier) or identifier
            )
            files = metadata.get("files", [])

            mp4_files = [f for f in files if f.get("name", "").endswith(".mp4")]
            if not mp4_files:
                mp4_files = [f for f in files if f.get("format") == "MPEG4"]
            if not mp4_files:
                if _check_ytdlp():
                    fallback = YtDlpDownloader(str(self.cache_dir.parent))
                    return fallback.download(url)
                return None

            video_file = min(mp4_files, key=lambda f: int(f.get("size", 0) or 0) * -1)
            if int(video_file.get("size", 0)) > 500 * 1024 * 1024:
                video_file = sorted(mp4_files, key=lambda f: int(f.get("size", 0) or 0))[0]

            download_url = f"https://archive.org/download/{identifier}/{video_file['name']}"

            cache_path = self._get_cache_path(url)
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            dl_resp = self._session.get(download_url, timeout=300, stream=True)
            dl_resp.raise_for_status()
            with open(cache_path, "wb") as f:
                for chunk in dl_resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            subjects = metadata.get("metadata", {}).get("subject", [])
            tags = [subjects] if isinstance(subjects, str) else subjects
            genre = _classify_genre(title, tags)

            meta = {
                "path": str(cache_path), "url": url, "title": title,
                "duration": float(metadata.get("metadata", {}).get("runtime", 0) or 0),
                "genre": genre, "source": "archive",
                "tags": tags,
            }
            self._save_meta(cache_path, meta)
            logger.info(f"[archive] Downloaded: {title[:60]} -> {cache_path}")
            return meta

        except Exception as e:
            logger.error(f"[archive] Error: {e}")
            if _check_ytdlp():
                fallback = YtDlpDownloader(str(self.cache_dir.parent))
                return fallback.download(url)
            return None

    def search(self, query: str, count: int = 5, **kwargs) -> List[str]:
        try:
            search_url = "https://archive.org/advancedsearch.php"
            resp = self._session.get(search_url, params={
                "q": f'mediatype:movies AND ({query})',
                "fl[]": "identifier,title",
                "rows": min(count, 50),
                "output": "json",
            }, timeout=30)
            resp.raise_for_status()
            docs = resp.json().get("response", {}).get("docs", [])
            urls = [f"https://archive.org/details/{d['identifier']}" for d in docs]
            return urls[:count]
        except Exception as e:
            logger.warning(f"[archive] Search error: {e}")
            return []

    def _extract_identifier(self, url: str) -> Optional[str]:
        m = re.search(r'archive\.org/details/([^/?#]+)', url)
        return m.group(1) if m else None

    def _load_cached_meta(self, cache_path: Path, url: str) -> Optional[Dict]:
        meta_path = cache_path.with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"path": str(cache_path), "url": url, "title": cache_path.stem,
                "duration": 0, "genre": "vlog", "source": "archive"}


def get_downloader(url: str = "", source: str = "", api_keys: Optional[Dict[str, str]] = None,
                   cache_dir: str = "") -> Optional[VideoSource]:
    """
    Auto-detect the right downloader for a URL or source name.
    Returns None if no downloader matches.
    """
    api_keys = api_keys or {}
    cd = cache_dir or str(DEFAULT_CACHE_DIR)

    downloaders: List[VideoSource] = [
        PexelsDownloader(api_keys.get("pexels", ""), cd),
        PixabayDownloader(api_keys.get("pixabay", ""), cd),
        InternetArchiveDownloader(cd),
        YtDlpDownloader(cd),
    ]

    source_lower = source.lower()
    if source_lower == "pexels":
        return downloaders[0]
    if source_lower == "pixabay":
        return downloaders[1]
    if source_lower in ("archive", "internetarchive"):
        return downloaders[2]
    if source_lower == "ytdlp":
        return downloaders[3]

    for d in downloaders:
        if d.supports(url):
            return d

    return downloaders[3]


def download_videos(
    urls: Optional[List[str]] = None,
    query: str = "",
    source: str = "pexels",
    count: int = 5,
    api_keys: Optional[Dict[str, str]] = None,
    cache_dir: str = "",
) -> Tuple[Dict[str, List[str]], List[Dict]]:
    """
    Download videos from URLs or search query.

    Returns:
        video_paths: {genre: [path1, path2, ...]}
        metas: [{path, url, title, duration, genre, source}, ...]
    """
    api_keys = api_keys or {}
    cd = cache_dir or str(DEFAULT_CACHE_DIR)
    metas: List[Dict] = []

    if not urls and query:
        dl = get_downloader(source=source, api_keys=api_keys, cache_dir=cd)
        if not dl:
            logger.error(f"No downloader available for source: {source}")
            return {}, []
        urls = dl.search(query, count)
        logger.info(f"Found {len(urls)} videos for query '{query}' from {source}")
        if not urls:
            return {}, []

    if not urls:
        return {}, []

    for url in urls:
        dl = get_downloader(url=url, source=source, api_keys=api_keys, cache_dir=cd)
        if not dl:
            logger.warning(f"No downloader for: {url}")
            continue

        meta = dl.download(url)
        if meta:
            metas.append(meta)
        else:
            logger.warning(f"Failed to download: {url}")

    video_paths: Dict[str, List[str]] = {}
    for meta in metas:
        genre = meta.get("genre", "vlog")
        video_paths.setdefault(genre, []).append(meta["path"])

    logger.info(f"Downloaded {len(metas)} videos across {len(video_paths)} genres: "
                f"{dict((g, len(p)) for g, p in video_paths.items())}")

    return video_paths, metas


def download_single(url: str, api_keys: Optional[Dict[str, str]] = None,
                    cache_dir: str = "") -> Optional[Dict]:
    """Convenience: download one video and return its meta."""
    dl = get_downloader(url=url, api_keys=api_keys, cache_dir=cache_dir)
    if not dl:
        return None
    return dl.download(url)
