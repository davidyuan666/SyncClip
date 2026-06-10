"""
Central configuration for SyncCLIPAgent experiments.
All paper parameters defined here with defaults from validation.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import json
import os

_VALIDATION_SPLIT_RATIO = 0.7

DEFAULT_MODEL_CONFIG: Dict = {
    "clip_model": "ViT-B/32",
    "clip_embedding_dim": 768,
    "whisper_model": "large-v3",
    "whisper_embedding_dim": 1280,
    "whisper_sample_rate": 16000,
    "llm_model": "gpt-4-mini",
    "gpt_model": "gpt-4o-mini",
    "common_projection_dim": 256,
    "projection_method": "pca_linear",
    "tts_model": "eleven_multilingual_v2",
    "tts_sample_rate": 8000,
}

DEFAULT_PARAMETER_SETTINGS: Dict[str, float] = {
    "theta": 0.65,
    "tau": 0.80,
    "alpha": 1.0,
    "beta": 0.5,
    "delta": 2.0,
}

DEFAULT_FPS_OPTIONS: List[int] = [1, 2, 3, 5]

DEFAULT_VIDEO_LENGTH_OPTIONS: List[int] = [5, 10, 15, 30]

DEFAULT_HARDWARE_CONFIG: Dict = {
    "gpu": "NVIDIA RTX 4090",
    "gpu_memory_gb": 25.2,
    "cpu": "AMD EPYC 9354",
    "cpu_cores": 16,
    "ram_gb": 60.1,
    "cuda_version": "12.1",
    "pytorch_version": "2.2.2",
}


@dataclass
class ExperimentConfig:
    model: Dict = field(default_factory=lambda: dict(DEFAULT_MODEL_CONFIG))
    parameters: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PARAMETER_SETTINGS))
    fps_options: List[int] = field(default_factory=lambda: list(DEFAULT_FPS_OPTIONS))
    video_length_options: List[int] = field(default_factory=lambda: list(DEFAULT_VIDEO_LENGTH_OPTIONS))
    hardware: Dict = field(default_factory=lambda: dict(DEFAULT_HARDWARE_CONFIG))

    dataset_dir: str = "experiments/data"
    output_dir: str = "experiments/output"
    annotation_dir: str = "experiments/annotations"
    seed: int = 42
    n_annotators: int = 3
    n_bootstrap: int = 1000
    confidence_level: float = 0.95
    mock_mode: bool = False
    use_gpu: bool = True
    openai_api_key: Optional[str] = None
    cos_enabled: bool = False
    tts_enabled: bool = False

    genre_list: List[str] = field(default_factory=lambda: [
        "action", "documentary", "vlog", "news", "sports", "music_video", "short_film",
    ])
    genre_counts: Dict[str, int] = field(default_factory=lambda: {
        "action": 20, "documentary": 15, "vlog": 25, "news": 10,
        "sports": 15, "music_video": 10, "short_film": 5,
    })
    genre_durations: Dict[str, float] = field(default_factory=lambda: {
        "action": 10.5, "documentary": 12.0, "vlog": 8.5, "news": 9.0,
        "sports": 11.0, "music_video": 7.5, "short_film": 14.0,
    })

    robustness_cases: List[str] = field(default_factory=lambda: [
        "low_resolution", "noisy_audio", "fast_scene_change",
        "non_english", "music_heavy",
    ])

    sensitivity_sweeps: Dict[str, List[float]] = field(default_factory=lambda: {
        "theta": [0.30, 0.45, 0.55, 0.65, 0.75, 0.85],
        "tau": [0.50, 0.65, 0.75, 0.80, 0.90, 0.95],
        "alpha_beta_ratio": [0.25, 0.5, 1.0, 2.0, 4.0],
        "fps": [1, 2, 3, 5],
    })

    def __post_init__(self):
        self.openai_api_key = self.openai_api_key or os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            self.llm_enabled = True

    def to_dict(self) -> Dict:
        return {
            "model": self.model,
            "parameters": self.parameters,
            "fps_options": self.fps_options,
            "hardware": self.hardware,
            "seed": self.seed,
            "n_annotators": self.n_annotators,
            "genre_list": self.genre_list,
            "mock_mode": self.mock_mode,
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def load_config(**overrides) -> ExperimentConfig:
    cfg = ExperimentConfig()
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg
