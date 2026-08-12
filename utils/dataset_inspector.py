"""
NeuroForge Dataset Inspector
Profiles local dataset files (CSV, Parquet, JSON) and generates
a DatasetProfile with column stats, quality scores, and task inference.
"""
from __future__ import annotations
import os
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from models.schemas import DatasetProfile, ColumnProfile, TaskDomain


class DatasetInspector:
    """
    Async-friendly dataset profiler.
    Detects format, computes column statistics, infers ML task type,
    and assigns a data quality score.
    """

    SUPPORTED_FORMATS = {".csv", ".parquet", ".json", ".jsonl", ".tsv"}

    async def profile(
        self,
        path: str,
        target_column: Optional[str] = None,
        task_hint: Optional[str] = None,
    ) -> DatasetProfile:
        """
        Profile a local dataset file.
        Runs synchronous pandas in a thread pool to avoid blocking the event loop.
        """
        return await asyncio.get_event_loop().run_in_executor(
            None, self._profile_sync, path, target_column, task_hint
        )

    def _profile_sync(
        self,
        path: str,
        target_column: Optional[str],
        task_hint: Optional[str],
    ) -> DatasetProfile:
        try:
            import pandas as pd
            import numpy as np
        except ImportError:
            raise RuntimeError("pandas and numpy are required for dataset profiling.")

        fpath = Path(path)
        if not fpath.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        ext = fpath.suffix.lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {ext}. Supported: {self.SUPPORTED_FORMATS}")

        size_mb = fpath.stat().st_size / (1024 * 1024)

        # Load data
        if ext == ".csv":
            df = pd.read_csv(path, nrows=50000)  # Cap at 50k rows for profiling
            fmt = "csv"
        elif ext == ".tsv":
            df = pd.read_csv(path, sep="\t", nrows=50000)
            fmt = "tsv"
        elif ext == ".parquet":
            df = pd.read_parquet(path)
            if len(df) > 50000:
                df = df.sample(50000, random_state=42)
            fmt = "parquet"
        elif ext in (".json", ".jsonl"):
            df = pd.read_json(path, lines=(ext == ".jsonl"), nrows=50000)
            fmt = "json"
        else:
            df = pd.read_csv(path, nrows=50000)
            fmt = "csv"

        rows, cols = df.shape

        # Column profiles
        column_profiles = []
        for col in df.columns:
            series = df[col]
            dtype = str(series.dtype)
            null_pct = round(series.isna().mean() * 100, 2)
            unique_count = series.nunique()
            sample = series.dropna().head(5).tolist()

            stats = {}
            if pd.api.types.is_numeric_dtype(series):
                desc = series.describe()
                stats = {
                    "mean": round(float(desc.get("mean", 0)), 4),
                    "std": round(float(desc.get("std", 0)), 4),
                    "min": round(float(desc.get("min", 0)), 4),
                    "max": round(float(desc.get("max", 0)), 4),
                    "median": round(float(series.median()), 4),
                }

            column_profiles.append(ColumnProfile(
                name=col,
                dtype=dtype,
                null_pct=null_pct,
                unique_count=int(unique_count),
                sample_values=[str(s) for s in sample],
                stats=stats,
            ))

        # Infer target column
        suggested_target = target_column or self._infer_target(df)

        # Infer task type
        inferred_task = self._infer_task(df, suggested_target, task_hint)

        # Quality score
        quality_score = self._quality_score(df, column_profiles)

        # Recommendations
        recommendations = self._generate_recommendations(df, column_profiles, quality_score)

        return DatasetProfile(
            path=str(fpath.resolve()),
            format=fmt,
            rows=rows,
            columns=cols,
            size_mb=round(size_mb, 2),
            column_profiles=column_profiles,
            inferred_task=inferred_task,
            suggested_target=suggested_target,
            data_quality_score=quality_score,
            recommendations=recommendations,
        )

    def _infer_target(self, df) -> Optional[str]:
        """Heuristically identify the target column."""
        import pandas as pd
        lower_cols = {c.lower(): c for c in df.columns}
        common_targets = ["target", "label", "class", "y", "output", "result",
                         "outcome", "prediction", "score", "category"]
        for t in common_targets:
            if t in lower_cols:
                return lower_cols[t]
        # Last column as fallback
        return str(df.columns[-1]) if len(df.columns) > 0 else None

    def _infer_task(self, df, target_col: Optional[str], hint: Optional[str]) -> TaskDomain:
        import pandas as pd

        if hint:
            hint_lower = hint.lower()
            if "vision" in hint_lower or "image" in hint_lower:
                return TaskDomain.COMPUTER_VISION
            if "nlp" in hint_lower or "text" in hint_lower or "language" in hint_lower:
                return TaskDomain.NLP
            if "rl" in hint_lower or "reinforcement" in hint_lower:
                return TaskDomain.REINFORCEMENT_LEARNING
            if "time" in hint_lower or "series" in hint_lower or "sequence" in hint_lower:
                return TaskDomain.TIME_SERIES

        if target_col and target_col in df.columns:
            target = df[target_col]
            n_unique = target.nunique()
            if pd.api.types.is_numeric_dtype(target):
                if n_unique <= 20:
                    return TaskDomain.CLASSICAL_ML  # classification
                else:
                    return TaskDomain.CLASSICAL_ML  # regression
            else:
                return TaskDomain.CLASSICAL_ML  # classification

        # Check column names for clues
        col_str = " ".join(df.columns).lower()
        if any(k in col_str for k in ["pixel", "image", "img"]):
            return TaskDomain.COMPUTER_VISION
        if any(k in col_str for k in ["text", "sentence", "word", "token"]):
            return TaskDomain.NLP
        if any(k in col_str for k in ["date", "time", "timestamp", "hour", "day"]):
            return TaskDomain.TIME_SERIES

        return TaskDomain.CLASSICAL_ML

    def _quality_score(self, df, profiles: List[ColumnProfile]) -> float:
        """Compute a data quality score 0–1."""
        scores = []

        # Null ratio score
        avg_null = sum(p.null_pct for p in profiles) / max(len(profiles), 1)
        null_score = max(0.0, 1.0 - avg_null / 100)
        scores.append(null_score)

        # Duplicate score
        dup_ratio = df.duplicated().mean()
        scores.append(max(0.0, 1.0 - dup_ratio))

        # Column diversity score (mixed types = better)
        import pandas as pd
        numeric_cols = len(df.select_dtypes(include="number").columns)
        categorical_cols = len(df.select_dtypes(include="object").columns)
        total = max(numeric_cols + categorical_cols, 1)
        diversity = min(numeric_cols, categorical_cols) / total
        scores.append(0.5 + diversity * 0.5)

        # Size adequacy score
        row_score = min(1.0, df.shape[0] / 1000)
        scores.append(row_score)

        return round(sum(scores) / len(scores), 3)

    def _generate_recommendations(
        self, df, profiles: List[ColumnProfile], quality_score: float
    ) -> List[str]:
        recs = []

        # High null columns
        high_null = [p.name for p in profiles if p.null_pct > 20]
        if high_null:
            recs.append(f"Columns with >20% nulls: {', '.join(high_null[:3])} — consider imputation or dropping")

        # Very high cardinality categoricals
        high_card = [p.name for p in profiles if p.dtype == "object" and p.unique_count > 100]
        if high_card:
            recs.append(f"High cardinality columns: {', '.join(high_card[:3])} — use embedding or target encoding")

        # Constant columns
        constant = [p.name for p in profiles if p.unique_count <= 1]
        if constant:
            recs.append(f"Constant columns detected: {', '.join(constant[:3])} — drop these")

        # Small dataset warning
        if df.shape[0] < 1000:
            recs.append("Small dataset (<1000 rows) — consider data augmentation, transfer learning, or few-shot approaches")

        # Large dataset suggestion
        if df.shape[0] > 100000:
            recs.append("Large dataset — consider mini-batch training with streaming DataLoader")

        if quality_score < 0.7:
            recs.append(f"Data quality score {quality_score:.2f} is low — prioritise data cleaning before training")

        if not recs:
            recs.append("Dataset looks clean and suitable for training")

        return recs
