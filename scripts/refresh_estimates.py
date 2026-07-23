"""Refresh static estimates embedded in committed benchmark result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from canifinetune import __version__
from canifinetune.bench.runner import BenchConfig, _build_estimate_breakdown


def refresh_result(path: Path) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    config = BenchConfig.model_validate(data["config"])
    gpu_total_gb = float(data.get("gpu", {}).get("total_vram_gb") or 0.0)
    if gpu_total_gb <= 0:
        return False
    breakdown = _build_estimate_breakdown(config, gpu_total_gb)
    data["estimated_total_gb"] = float(breakdown["total_estimated_gb"])
    data["estimated_breakdown"] = breakdown
    data["estimator_version"] = __version__
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results",
        nargs="?",
        type=Path,
        default=Path("benchmarks/results"),
    )
    args = parser.parse_args()
    files = sorted(args.results.glob("*.json"))
    changed = sum(refresh_result(path) for path in files)
    print(f"refreshed {changed} result file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
