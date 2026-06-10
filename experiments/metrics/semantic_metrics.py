"""
Semantic correspondence metrics (Eq. 17 in paper).
Computes cosine similarity between visual descriptions and audio transcripts.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np


@dataclass
class SemanticMetrics:
    mean_similarity: float = 0.0
    std_similarity: float = 0.0
    visual_avg: float = 0.0
    audio_avg: float = 0.0
    cross_modal_avg: float = 0.0
    per_genre: Dict[str, "SemanticMetrics"] = field(default_factory=dict)
    similarities: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        base = {
            "mean_similarity": self.mean_similarity,
            "std_similarity": self.std_similarity,
            "visual_avg": self.visual_avg,
            "audio_avg": self.audio_avg,
            "cross_modal_avg": self.cross_modal_avg,
        }
        if self.per_genre:
            base["per_genre"] = {g: m.to_dict() for g, m in self.per_genre.items()}
        return base

    def __repr__(self):
        return f"SemanticMetrics(mean={self.mean_similarity:.4f}, cross_modal={self.cross_modal_avg:.4f})"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a_norm, b_norm))


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    sim = cosine_similarity(a, b)
    return max(0.0, min(1.0, sim))


def compute_semantic_correspondence(
    visual_embeddings: List[np.ndarray],
    audio_embeddings: List[np.ndarray],
    modality_labels: Optional[List[str]] = None,
) -> SemanticMetrics:
    """
    Compute semantic correspondence between N visual and N audio embeddings.

    For each timestamp i, computes:
        sim(W(A_i), D(V_i))  where D(V_i) is the CLIP visual embedding
        and W(A_i) is the Whisper audio embedding (projected to same space).

    If embeddings are already projected to a shared space (d_c),
    cosine similarity is computed directly.
    """
    n = len(visual_embeddings)
    if n == 0:
        return SemanticMetrics()

    sims = []
    for i in range(min(n, len(audio_embeddings))):
        sim = _safe_cosine(visual_embeddings[i], audio_embeddings[i])
        sims.append(sim)

    sims_arr = np.array(sims)

    result = SemanticMetrics(
        mean_similarity=round(float(np.mean(sims_arr)), 4),
        std_similarity=round(float(np.std(sims_arr)), 4),
        visual_avg=round(float(np.mean([float(np.linalg.norm(v)) for v in visual_embeddings])), 4),
        audio_avg=round(float(np.mean([float(np.linalg.norm(a)) for a in audio_embeddings])), 4),
        cross_modal_avg=round(float(np.mean(sims_arr)), 4),
        similarities=[round(s, 4) for s in sims],
    )
    return result


def compute_genre_semantic_breakdown(
    visual_by_genre: Dict[str, List[np.ndarray]],
    audio_by_genre: Dict[str, List[np.ndarray]],
) -> Dict[str, SemanticMetrics]:
    return {
        genre: compute_semantic_correspondence(
            visual_by_genre.get(genre, []),
            audio_by_genre.get(genre, []),
        )
        for genre in visual_by_genre
    }
