"""
Cross-modal projection module.
Implements Algorithm 1 lines 8-9: projects CLIP (768-dim) and Whisper (1280-dim)
embeddings into a shared d_c=256-dim latent space for cosine similarity matching.

Methods: PCA + linear projection (default), with optional learned MLP.
"""
import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CrossModalProjection:
    """
    Projects CLIP visual (768-dim) and Whisper audio (1280-dim) embeddings
    into a common d_c-dimensional space.

    Z_v = norm(W_v @ Phi_v)     for visual (CLIP)
    Z_a = norm(W_a @ Phi_a)     for audio (Whisper)

    Where W_v and W_a are obtained via PCA dimensionality reduction followed
    by a linear projection layer.

    Reference: Eq. 8-9 in the SyncCLIPAgent paper.
    """

    def __init__(
        self,
        d_visual: int = 768,
        d_audio: int = 1280,
        d_common: int = 256,
        method: str = "pca_linear",
        random_seed: int = 42,
    ):
        self.d_visual = d_visual
        self.d_audio = d_audio
        self.d_common = d_common
        self.method = method
        self.rng = np.random.default_rng(random_seed)

        self.W_v: Optional[np.ndarray] = None  # (d_common, d_visual)
        self.W_a: Optional[np.ndarray] = None  # (d_common, d_audio)

        self.v_pca_mean: Optional[np.ndarray] = None
        self.v_pca_components: Optional[np.ndarray] = None
        self.a_pca_mean: Optional[np.ndarray] = None
        self.a_pca_components: Optional[np.ndarray] = None

        self.is_fitted = False

    def fit(
        self,
        visual_embeddings: np.ndarray,
        audio_embeddings: np.ndarray,
    ) -> "CrossModalProjection":
        """
        Fit projection matrices on training data.

        Args:
            visual_embeddings: (N_v, d_visual) CLIP embeddings
            audio_embeddings: (N_a, d_audio) Whisper embeddings
        """
        visual_embeddings = np.asarray(visual_embeddings, dtype=np.float64)
        audio_embeddings = np.asarray(audio_embeddings, dtype=np.float64)

        if self.method == "pca_linear":
            self._fit_pca_linear(visual_embeddings, audio_embeddings)
        elif self.method == "fixed_random":
            self._fit_fixed_random()
        elif self.method == "mlp_adapter":
            self._fit_mlp_adapter(visual_embeddings, audio_embeddings)
        else:
            raise ValueError(f"Unknown projection method: {self.method}")

        self.is_fitted = True
        logger.info(f"CrossModalProjection fitted with method={self.method}, d_c={self.d_common}")
        return self

    def _fit_pca_linear(self, V: np.ndarray, A: np.ndarray):
        V_centered, self.v_pca_mean, self.v_pca_components = _pca_fit(V, self.d_common)
        A_centered, self.a_pca_mean, self.a_pca_components = _pca_fit(A, self.d_common)

        self.W_v = self.v_pca_components.T  # (d_common, d_visual)
        self.W_a = self.a_pca_components.T  # (d_common, d_audio)

        self.W_v = self.W_v / (np.linalg.norm(self.W_v, axis=1, keepdims=True) + 1e-8)
        self.W_a = self.W_a / (np.linalg.norm(self.W_a, axis=1, keepdims=True) + 1e-8)

    def _fit_fixed_random(self):
        self.W_v = self.rng.normal(0, 1.0 / np.sqrt(self.d_visual), (self.d_common, self.d_visual))
        self.W_a = self.rng.normal(0, 1.0 / np.sqrt(self.d_audio), (self.d_common, self.d_audio))
        self.W_v = self.W_v / (np.linalg.norm(self.W_v, axis=1, keepdims=True) + 1e-8)
        self.W_a = self.W_a / (np.linalg.norm(self.W_a, axis=1, keepdims=True) + 1e-8)

    def _fit_mlp_adapter(self, V: np.ndarray, A: np.ndarray):
        raise NotImplementedError("MLP adapter requires torch and training loop. Use pca_linear instead.")

    def project_visual(self, embedding: np.ndarray) -> np.ndarray:
        """Project CLIP visual embedding to d_c-dim and L2-normalize."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before project_visual()")
        x = np.asarray(embedding, dtype=np.float64)
        if self.method == "pca_linear" and self.v_pca_mean is not None:
            x = x - self.v_pca_mean
        if x.ndim == 1:
            z = self.W_v @ x
        else:
            z = (self.W_v @ x.T).T
        return _l2_normalize(z)

    def project_audio(self, embedding: np.ndarray) -> np.ndarray:
        """Project Whisper audio embedding to d_c-dim and L2-normalize."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before project_audio()")
        x = np.asarray(embedding, dtype=np.float64)
        if self.method == "pca_linear" and self.a_pca_mean is not None:
            x = x - self.a_pca_mean
        if x.ndim == 1:
            z = self.W_a @ x
        else:
            z = (self.W_a @ x.T).T
        return _l2_normalize(z)

    def compute_alignment_matrix(self, V: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Compute cross-modal alignment probability matrix: softmax(Z_v @ Z_a^T / sqrt(d_c))."""
        Z_v = self.project_visual(V)
        Z_a = self.project_audio(A)
        if Z_v.ndim == 1:
            Z_v = Z_v.reshape(1, -1)
        if Z_a.ndim == 1:
            Z_a = Z_a.reshape(1, -1)
        logits = Z_v @ Z_a.T / np.sqrt(self.d_common)
        return _stable_softmax(logits, axis=1)

    def get_alignment_pairs(self, V: np.ndarray, A: np.ndarray, tau: float = 0.8) -> list:
        """Return alignment pairs where P_ij > tau (Eq. from Algorithm 1 line 11)."""
        P = self.compute_alignment_matrix(V, A)
        pairs = []
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                if P[i, j] > tau:
                    pairs.append((int(i), int(j), float(P[i, j])))
        return pairs

    def to_dict(self) -> Dict:
        return {
            "method": self.method,
            "d_visual": self.d_visual,
            "d_audio": self.d_audio,
            "d_common": self.d_common,
            "is_fitted": self.is_fitted,
            "W_v_shape": list(self.W_v.shape) if self.W_v is not None else None,
            "W_a_shape": list(self.W_a.shape) if self.W_a is not None else None,
        }


def _pca_fit(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple PCA: center, compute covariance, get top eigenvectors."""
    mean = X.mean(axis=0)
    X_centered = X - mean
    cov = X_centered.T @ X_centered / (X_centered.shape[0] - 1)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    top_components = eigenvectors[:, idx[:n_components]]

    return X_centered, mean, top_components


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x / (np.linalg.norm(x) + 1e-8)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def _stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / (np.sum(e_x, axis=axis, keepdims=True) + 1e-8)
