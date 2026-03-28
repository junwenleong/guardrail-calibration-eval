"""PlotGenerator: publication-quality calibration visualizations.

All figures use colorblind-safe palettes and are saved as PDF.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Colorblind-safe palette (Wong 2011)
_COLORS = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#000000"
]


class PlotGenerator:
    """Generates publication-quality calibration plots and tables."""

    def __init__(self, output_dir: str | Path = "results"):
        self.output_dir = Path(output_dir)
        (self.output_dir / "figures").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "metrics").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "predictions").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # ECE vs shift level
    # ------------------------------------------------------------------

    def plot_ece_vs_shift(
        self,
        results: dict,  # {guardrail_name: {axis: {shift_level: ECEResult}}}
        axis: int,
        save: bool = True,
    ):
        """ECE vs shift_level, one line per guardrail, 95% CI bands.

        Publication-quality vector format (PDF), colorblind-safe palette.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (guardrail, axis_results) in enumerate(results.items()):
            if axis not in axis_results:
                continue
            shift_data = axis_results[axis]
            shift_levels = sorted(shift_data.keys())
            eces = [shift_data[sl].ece for sl in shift_levels]
            ci_lowers = [shift_data[sl].ci_lower for sl in shift_levels]
            ci_uppers = [shift_data[sl].ci_upper for sl in shift_levels]

            color = _COLORS[i % len(_COLORS)]
            ax.plot(shift_levels, eces, marker="o", label=guardrail, color=color)
            ax.fill_between(shift_levels, ci_lowers, ci_uppers, alpha=0.2, color=color)

        ax.set_xlabel("Shift Level")
        ax.set_ylabel("ECE")
        ax.set_title(f"ECE vs Distribution Shift — Axis {axis}")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_ylim(0, 1)
        plt.tight_layout()

        if save:
            path = self.output_dir / "figures" / f"ece_vs_shift_axis{axis}.pdf"
            fig.savefig(path, format="pdf", bbox_inches="tight")
            logger.info("Saved: %s", path)
        return fig

    # ------------------------------------------------------------------
    # Reliability heatmap
    # ------------------------------------------------------------------

    def plot_reliability_heatmap(
        self,
        ece_matrix: dict[str, dict[int, float]],  # {guardrail: {axis: ece}}
        save: bool = True,
    ):
        """Guardrails × Axes heatmap with ECE as color."""
        import matplotlib.pyplot as plt

        guardrails = sorted(ece_matrix.keys())
        axes = sorted({ax for g in ece_matrix.values() for ax in g.keys()})

        data = np.array([
            [ece_matrix[g].get(ax, np.nan) for ax in axes]
            for g in guardrails
        ])

        fig, ax = plt.subplots(figsize=(8, max(4, len(guardrails))))
        im = ax.imshow(data, cmap="RdYlGn_r", vmin=0, vmax=0.5, aspect="auto")
        plt.colorbar(im, ax=ax, label="ECE")

        ax.set_xticks(range(len(axes)))
        ax.set_xticklabels([f"Axis {a}" for a in axes])
        ax.set_yticks(range(len(guardrails)))
        ax.set_yticklabels(guardrails)
        ax.set_title("Calibration Heatmap (ECE by Guardrail × Axis)")

        # Annotate cells
        for i in range(len(guardrails)):
            for j in range(len(axes)):
                val = data[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)

        plt.tight_layout()
        if save:
            path = self.output_dir / "figures" / "reliability_heatmap.pdf"
            fig.savefig(path, format="pdf", bbox_inches="tight")
            logger.info("Saved: %s", path)
        return fig

    # ------------------------------------------------------------------
    # Safety risk heatmap (EOE + honest threshold overlay)
    # ------------------------------------------------------------------

    def plot_safety_risk_heatmap(
        self,
        eoe_matrix: dict[str, dict[int, float]],
        threshold_matrix: dict[str, dict[int, float]],
        save: bool = True,
    ):
        """Overlay EOE onto honest threshold table.

        Rows: guardrails, Columns: axes.
        Cell color: EOE value. Cell annotation: honest threshold.

        Shows practitioners not just whether a model is miscalibrated,
        but whether it is miscalibrated in the dangerous direction
        (false negatives / overconfident benign predictions).
        """
        import matplotlib.pyplot as plt

        guardrails = sorted(eoe_matrix.keys())
        axes = sorted({ax for g in eoe_matrix.values() for ax in g.keys()})

        eoe_data = np.array([
            [eoe_matrix[g].get(ax, np.nan) for ax in axes]
            for g in guardrails
        ])
        threshold_data = np.array([
            [threshold_matrix[g].get(ax, np.nan) for ax in axes]
            for g in guardrails
        ])

        fig, ax = plt.subplots(figsize=(8, max(4, len(guardrails))))
        im = ax.imshow(eoe_data, cmap="Reds", vmin=0, vmax=0.3, aspect="auto")
        plt.colorbar(im, ax=ax, label="EOE (Expected Overconfidence Error)")

        ax.set_xticks(range(len(axes)))
        ax.set_xticklabels([f"Axis {a}" for a in axes])
        ax.set_yticks(range(len(guardrails)))
        ax.set_yticklabels(guardrails)
        ax.set_title(
            "Safety Risk Heatmap\n"
            "(Color: EOE, Annotation: Honest Threshold @ 90% Precision)"
        )

        for i in range(len(guardrails)):
            for j in range(len(axes)):
                eoe_val = eoe_data[i, j]
                thr_val = threshold_data[i, j]
                if not np.isnan(eoe_val):
                    label = f"EOE={eoe_val:.3f}"
                    if not np.isnan(thr_val):
                        label += f"\nτ={thr_val:.2f}"
                    ax.text(j, i, label, ha="center", va="center", fontsize=7)

        plt.tight_layout()
        if save:
            path = self.output_dir / "figures" / "safety_risk_heatmap.pdf"
            fig.savefig(path, format="pdf", bbox_inches="tight")
            logger.info("Saved: %s", path)
        return fig

    # ------------------------------------------------------------------
    # Honest threshold table
    # ------------------------------------------------------------------

    def generate_honest_threshold_table(
        self,
        thresholds: list,  # list[HonestThreshold]
        save: bool = True,
    ) -> pd.DataFrame:
        """Per-guardrail per-axis honest thresholds with bootstrap CI.

        Logit-based and api_score in separate sections.
        Includes conservative threshold (CI lower bound).
        """
        rows = []
        for ht in thresholds:
            rows.append({
                "guardrail": ht.guardrail,
                "axis": ht.axis,
                "shift_level": ht.shift_level,
                "target_precision": ht.target_precision,
                "honest_threshold": ht.honest_confidence,
                "conservative_threshold": ht.ci_lower,  # CI lower bound
                "ci_upper": ht.ci_upper,
                "actual_precision": ht.actual_precision_at_target,
            })

        df = pd.DataFrame(rows)
        if save and not df.empty:
            path = self.output_dir / "metrics" / "honest_thresholds.csv"
            df.to_csv(path, index=False)
            logger.info("Saved: %s", path)
        return df

    # ------------------------------------------------------------------
    # APAC language comparison
    # ------------------------------------------------------------------

    def plot_apac_language_comparison(
        self,
        results: dict,  # {language: {guardrail: {metric: value}}}
        save: bool = True,
    ):
        """ECE, Brier, precision-at-threshold across EN/MS/ZH/ID."""
        import matplotlib.pyplot as plt

        languages = ["en", "ms", "zh", "id"]
        guardrails = sorted({
            g for lang_data in results.values() for g in lang_data.keys()
        })
        metrics = ["ece", "brier_score"]

        fig, axes = plt.subplots(1, len(metrics), figsize=(12, 5))
        for ax_idx, metric in enumerate(metrics):
            ax = axes[ax_idx]
            x = np.arange(len(languages))
            width = 0.8 / max(len(guardrails), 1)

            for i, guardrail in enumerate(guardrails):
                values = [
                    results.get(lang, {}).get(guardrail, {}).get(metric, np.nan)
                    for lang in languages
                ]
                offset = (i - len(guardrails) / 2) * width + width / 2
                ax.bar(x + offset, values, width, label=guardrail,
                       color=_COLORS[i % len(_COLORS)], alpha=0.8)

            ax.set_xticks(x)
            ax.set_xticklabels(languages)
            ax.set_ylabel(metric.upper())
            ax.set_title(f"{metric.upper()} by Language")
            ax.legend(fontsize=7)

        plt.suptitle("APAC Language Calibration Comparison")
        plt.tight_layout()
        if save:
            path = self.output_dir / "figures" / "apac_language_comparison.pdf"
            fig.savefig(path, format="pdf", bbox_inches="tight")
            logger.info("Saved: %s", path)
        return fig

    # ------------------------------------------------------------------
    # Bin sensitivity sweep
    # ------------------------------------------------------------------

    def plot_bin_sensitivity_sweep(
        self,
        sweep_results: list[tuple[int, float]],
        adaptive_m: int,
        guardrail: str = "",
        save: bool = True,
    ):
        """Line plot of ECE vs M for the bin sensitivity sweep.

        Shows the adaptive formula's M as a vertical marker.
        """
        import matplotlib.pyplot as plt

        ms = [r[0] for r in sweep_results]
        eces = [r[1] for r in sweep_results]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(ms, eces, marker="o", color=_COLORS[0], label="ECE")
        ax.axvline(x=adaptive_m, color="red", linestyle="--",
                   label=f"Adaptive M={adaptive_m}")
        ax.set_xlabel("Number of Bins (M)")
        ax.set_ylabel("ECE")
        ax.set_title(f"Bin Sensitivity Sweep — {guardrail}")
        ax.legend()
        plt.tight_layout()

        if save:
            name = guardrail.replace(" ", "_").replace("(", "").replace(")", "")
            path = self.output_dir / "figures" / f"bin_sensitivity_{name}.pdf"
            fig.savefig(path, format="pdf", bbox_inches="tight")
            logger.info("Saved: %s", path)
        return fig

    # ------------------------------------------------------------------
    # Result persistence
    # ------------------------------------------------------------------

    def persist_predictions(
        self,
        predictions: list,  # list[Prediction]
        guardrail_name: str,
    ) -> None:
        """Persist predictions to Parquet (primary) + CSV (backup)."""
        df = pd.DataFrame([asdict(p) for p in predictions])
        name = guardrail_name.replace(" ", "_").replace("(", "").replace(")", "")

        parquet_path = self.output_dir / "predictions" / f"{name}.parquet"
        csv_path = self.output_dir / "predictions" / f"{name}.csv"

        df.to_parquet(parquet_path, index=False)
        df.to_csv(csv_path, index=False)
        logger.info("Persisted %d predictions for %s", len(predictions), guardrail_name)

    def persist_metrics(self, metrics: dict[str, Any], name: str) -> None:
        """Persist metrics dict to JSON."""
        path = self.output_dir / "metrics" / f"{name}.json"
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info("Persisted metrics: %s", path)

    def log_run_metadata(
        self,
        model_versions: dict[str, str],
        quantization_settings: dict[str, str],
        random_seed: int,
    ) -> None:
        """Log exact model versions, quantization settings, and random seeds."""
        metadata = {
            "model_versions": model_versions,
            "quantization_settings": quantization_settings,
            "random_seed": random_seed,
        }
        self.persist_metrics(metadata, "run_metadata")
        logger.info("Run metadata logged: seed=%d", random_seed)
