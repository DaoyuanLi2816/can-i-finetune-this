"""Compile and lint actual rendered Python for each supported training method."""

import subprocess
import sys
import tempfile
from pathlib import Path

from canifinetune.recipes import RecipeRequest, generate_recipe

with tempfile.TemporaryDirectory() as temporary:
    for method in ("full", "lora", "qlora"):
        out = Path(temporary) / method
        generate_recipe(
            RecipeRequest(model_id="Qwen/Qwen2.5-0.5B-Instruct", method=method, output_dir=out)
        )
        for path in out.glob("*.py"):
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ruff",
                    "check",
                    "--isolated",
                    "--select",
                    "E,F,B",
                    str(path),
                ],
                check=True,
            )
