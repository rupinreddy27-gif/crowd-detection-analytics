"""Occupancy analytics: people counts over time and summary statistics."""

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd


@dataclass
class OccupancyTracker:
    """Records how many people were seen in each processed frame.

    capacity: optional limit; the summary reports how long the count exceeded it.
    smoothing_window: frames used for a rolling average, which hides one-frame
        detection flickers.
    """

    capacity: Optional[int] = None
    smoothing_window: int = 5
    _rows: List[dict] = field(default_factory=list)

    def record(self, frame_index: int, timestamp_s: float, count: int) -> None:
        self._rows.append({"frame": frame_index, "time_s": round(timestamp_s, 3), "count": int(count)})

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self._rows, columns=["frame", "time_s", "count"])
        if not df.empty:
            df["smoothed"] = df["count"].rolling(self.smoothing_window, min_periods=1).mean().round(2)
        else:
            df["smoothed"] = pd.Series(dtype=float)
        return df

    def summary(self) -> dict:
        df = self.to_dataframe()
        if df.empty:
            return {"frames_analyzed": 0}

        peak_idx = df["count"].idxmax()
        summary = {
            "frames_analyzed": int(len(df)),
            "duration_s": float(df["time_s"].iloc[-1] - df["time_s"].iloc[0]),
            "average_count": round(float(df["count"].mean()), 2),
            "median_count": float(df["count"].median()),
            "peak_count": int(df.loc[peak_idx, "count"]),
            "peak_time_s": float(df.loc[peak_idx, "time_s"]),
            "min_count": int(df["count"].min()),
        }

        if self.capacity is not None:
            over = df["count"] > self.capacity
            # Each sampled frame stands for the time until the next sample.
            step = df["time_s"].diff().shift(-1).fillna(df["time_s"].diff().median() if len(df) > 1 else 0)
            summary["capacity"] = int(self.capacity)
            summary["frames_over_capacity"] = int(over.sum())
            summary["seconds_over_capacity"] = round(float(step[over].sum()), 2)
            summary["pct_time_over_capacity"] = round(100 * float(over.mean()), 2)
        return summary

    def save_csv(self, path) -> None:
        self.to_dataframe().to_csv(path, index=False)

    def save_chart(self, path, title: str = "People count over time") -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = self.to_dataframe()
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(df["time_s"], df["count"], alpha=0.35, linewidth=1, label="Per frame")
        ax.plot(df["time_s"], df["smoothed"], linewidth=2, label=f"Rolling avg ({self.smoothing_window} frames)")
        if self.capacity is not None:
            ax.axhline(self.capacity, color="red", linestyle="--", linewidth=1, label=f"Capacity ({self.capacity})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("People detected")
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
