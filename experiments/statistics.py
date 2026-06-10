"""
Statistical analysis: bootstrap confidence intervals, significance tests,
and LaTeX table formatter for manuscript integration.
"""
import json
import os
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ConfidenceInterval:
    metric_name: str
    mean: float
    lower: float
    upper: float
    confidence_level: float = 0.95

    def __repr__(self):
        return f"{self.metric_name}: {self.mean:.4f} [{self.lower:.4f}, {self.upper:.4f}]"

    def to_latex(self) -> str:
        return f"${self.mean:.3f}$ [{self.lower:.3f}, {self.upper:.3f}]"


@dataclass
class TTestResult:
    metric_name: str
    t_statistic: float
    p_value: float
    significant: bool
    alpha: float = 0.05

    def __repr__(self):
        sig = "*" if self.significant else "ns"
        return f"{self.metric_name}: t={self.t_statistic:.3f}, p={self.p_value:.4f} {sig}"


class StatisticsAnalyzer:
    def __init__(self, seed: int = 42, n_bootstrap: int = 1000):
        self.rng = np.random.default_rng(seed)
        self.n_bootstrap = n_bootstrap

    def bootstrap_ci(
        self,
        data: np.ndarray,
        confidence: float = 0.95,
        metric_name: str = "metric",
    ) -> ConfidenceInterval:
        if len(data) == 0:
            return ConfidenceInterval(metric_name=metric_name, mean=0, lower=0, upper=0)

        n = len(data)
        means = np.zeros(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            sample = self.rng.choice(data, size=n, replace=True)
            means[i] = np.mean(sample)

        alpha = 1 - confidence
        lower = np.percentile(means, 100 * alpha / 2)
        upper = np.percentile(means, 100 * (1 - alpha / 2))

        return ConfidenceInterval(
            metric_name=metric_name,
            mean=round(float(np.mean(means)), 4),
            lower=round(float(lower), 4),
            upper=round(float(upper), 4),
            confidence_level=confidence,
        )

    def welch_ttest(
        self,
        a: np.ndarray,
        b: np.ndarray,
        metric_name: str = "metric",
        alpha: float = 0.05,
    ) -> TTestResult:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)

        m_a, m_b = np.mean(a), np.mean(b)
        v_a, v_b = np.var(a, ddof=1), np.var(b, ddof=1)
        n_a, n_b = len(a), len(b)

        if v_a == 0 and v_b == 0:
            return TTestResult(metric_name=metric_name, t_statistic=0, p_value=1.0, significant=False)

        se = np.sqrt(v_a / n_a + v_b / n_b)
        if se < 1e-10:
            return TTestResult(metric_name=metric_name, t_statistic=0, p_value=1.0, significant=False)

        t_stat = (m_a - m_b) / se

        df_num = (v_a / n_a + v_b / n_b) ** 2
        df_den = (v_a / n_a) ** 2 / (n_a - 1) + (v_b / n_b) ** 2 / (n_b - 1)
        df = df_num / df_den if df_den > 1e-10 else 1.0

        p_value = float(2 * _t_survival(abs(t_stat), df))

        return TTestResult(
            metric_name=metric_name,
            t_statistic=round(float(t_stat), 4),
            p_value=round(p_value, 4),
            significant=p_value < alpha,
            alpha=alpha,
        )

    def cohens_kappa_from_confusion(self, confusion: np.ndarray) -> float:
        n = np.sum(confusion)
        if n == 0:
            return 0.0
        p_o = np.trace(confusion) / n
        row_sums = confusion.sum(axis=1)
        col_sums = confusion.sum(axis=0)
        p_e = np.sum(row_sums * col_sums) / (n * n) if n > 0 else 0.0
        return (p_o - p_e) / (1 - p_e) if p_e < 1 else 0.0


class LaTeXTableGenerator:
    """Generate LaTeX tables matching paper format for manuscript integration."""

    def __init__(self, output_dir: str = "experiments/output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def segment_accuracy_table(
        self,
        per_genre_fps: Dict[str, Dict[int, Dict[str, float]]],
    ) -> str:
        genre_order = ["Action", "Documentary", "Vlog", "News", "Sports", "Music Video"]
        fps_order = [1, 2, 3, 5]

        lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Multimodal Segment Selection Accuracy Analysis\label{tab:segment_accuracy}}",
            r"\small",
            r"\renewcommand{\arraystretch}{1.2}",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{@{}l *{4}{S[table-format=1.2]S[table-format=1.2]S[table-format=1.2]} @{}}",
            r"\toprule[1.5pt]",
            r"\multirow{2}{*}{\textbf{Genre}} & ",
        ]

        fps_headers = []
        for fps in fps_order:
            fps_headers.append(
                r"\multicolumn{3}{c}{\textbf{" + f"{fps} fps" + r"}}"
            )
        lines.append(" & ".join(fps_headers) + r" \\")
        lines.append(
            " & ".join([r"\cmidrule(lr){" + f"{2+i*3}-{4+i*3}" + r"}" for i in range(len(fps_order))]) + r" \\"
        )

        sub_headers = " & " + " & ".join([r"{P} & {R} & {F1}" for _ in fps_order])
        lines.append(sub_headers + r" \\")
        lines.append(r"\midrule[0.8pt]")

        for genre in genre_order:
            key = genre.lower().replace(" ", "_")
            if key not in per_genre_fps:
                continue
            row_parts = [genre]
            for fps in fps_order:
                metrics = per_genre_fps[key].get(fps, {"P": 0, "R": 0, "F1": 0})
                row_parts.append(f"{metrics.get('P', 0):.2f}")
                row_parts.append(f"{metrics.get('R', 0):.2f}")
                row_parts.append(f"{metrics.get('F1', 0):.2f}")
            lines.append(" & ".join(row_parts) + r" \\")
            if genre in ("Documentary", "News"):
                lines.append(r"\addlinespace[0.2em]")

        lines.append(r"\addlinespace[0.5em]")
        lines.append(r"\midrule[0.8pt]")

        avg_row = [r"\textbf{Avg.}"]
        for fps in fps_order:
            p_vals = []
            r_vals = []
            f1_vals = []
            for genre in genre_order:
                key = genre.lower().replace(" ", "_")
                m = per_genre_fps.get(key, {}).get(fps, {})
                p_vals.append(m.get("P", 0))
                r_vals.append(m.get("R", 0))
                f1_vals.append(m.get("F1", 0))
            avg_row.append(f"{np.mean(p_vals):.2f}")
            avg_row.append(f"{np.mean(r_vals):.2f}")
            avg_row.append(f"{np.mean(f1_vals):.2f}")
        lines.append(" & ".join(avg_row) + r" \\")
        lines.append(r"\bottomrule[1.5pt]")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table*}")

        return "\n".join(lines)

    def performance_comparison_table(
        self,
        methods_data: List[Dict],
    ) -> str:
        lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Cross-Modal Audio-Visual Synchronization Performance\label{tab:performance_comparison}}",
            r"\small",
            r"\renewcommand{\arraystretch}{1.2}",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{@{}l *{3}{S[table-format=1.2]S[table-format=1.2]S[table-format=1.2]} @{}}",
            r"\toprule[1.5pt]",
            r"\multirow{2}{*}{\textbf{Method}} & ",
            r"\multicolumn{3}{c}{\textbf{System Performance}} & ",
            r"\multicolumn{3}{c}{\textbf{User Experience}} & ",
            r"\multicolumn{3}{c@{}}{\textbf{Composite Score}} \\",
            r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(l){8-10}",
            r" & {P} & {F1} & {Time} & {Sat.} & {Eff.} & {Use} & {V} & {D} & {N} \\",
            r"\midrule[0.8pt]",
        ]

        for method in methods_data:
            parts = [method["name"]]
            parts.append(f"{method.get('P', 0):.2f}")
            parts.append(f"{method.get('F1', 0):.2f}")
            parts.append(f"{method.get('Time', 0):.1f}")
            parts.append(f"{method.get('Sat', 0):.1f}")
            parts.append(f"{method.get('Eff', 0):.1f}")
            parts.append(f"{method.get('Use', 0):.1f}")
            parts.append(f"{method.get('V', 0):.2f}")
            parts.append(f"{method.get('D', 0):.2f}")
            parts.append(f"{method.get('N', 0):.2f}")
            lines.append(" & ".join(parts) + r" \\")
            if method["name"] == "SVM-Based":
                lines.append(r"\addlinespace[0.2em]")
            if method["name"] == "AV-Summary":
                lines.append(r"\addlinespace[0.2em]")

        lines.append(r"\addlinespace[0.5em]")
        lines.append(r"\midrule[0.8pt]")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table*}")

        return "\n".join(lines)

    def runtime_table(self, timings: List[Dict]) -> str:
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Per-Component Runtime and Resource Report\label{tab:runtime_result}}",
            r"\small",
            r"\renewcommand{\arraystretch}{1.15}",
            r"\begin{tabular}{@{}p{0.30\linewidth}p{0.24\linewidth}p{0.24\linewidth}@{}}",
            r"\toprule",
            r"Component & Runtime (sec/video-min) & GPU Memory (GB) \\",
            r"\midrule",
        ]

        for t in timings:
            comp = t.get("component", "").replace("_", " ").title()
            runtime = f"{t.get('mean_sec_per_video_min', 0):.1f} ± {t.get('std_sec_per_video_min', 0):.1f}"
            gpu = f"{t.get('peak_gpu_memory_gb', 'N/A'):.1f}" if t.get("peak_gpu_memory_gb") else "N/A"
            lines.append(f"{comp} & {runtime} & {gpu} \\\\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        return "\n".join(lines)

    def save_all(self, tables: Dict[str, str]) -> Dict[str, str]:
        saved = {}
        for name, content in tables.items():
            path = os.path.join(self.output_dir, f"{name}.tex")
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            saved[name] = path
        return saved


def _t_survival(t: float, df: float, steps: int = 1000) -> float:
    """Approximate t-distribution survival function via Simpson integration."""
    import math

    if t < 0:
        return 1 - _t_survival(-t, df)
    if df <= 0:
        return 0.5

    a_norm = math.lgamma((df + 1) / 2) - math.lgamma(df / 2)
    b_norm = 0.5 * math.log(df * math.pi)
    log_norm = a_norm - b_norm

    def f(x):
        return math.exp(log_norm - (df + 1) / 2 * math.log(1 + x * x / df))

    h = t / steps
    total = f(0) + f(t)
    for i in range(1, steps):
        x = i * h
        total += 4 * f(x) if i % 2 == 1 else 2 * f(x)

    p_right = h * total / 3
    return min(0.5, max(0, p_right))
