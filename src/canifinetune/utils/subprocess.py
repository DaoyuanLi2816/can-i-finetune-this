"""Subprocess helpers that swallow errors and never raise into the CLI."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence


def try_run(cmd: Sequence[str], timeout: float = 10.0) -> tuple[int, str, str]:
    """Run ``cmd`` and return ``(returncode, stdout, stderr)``.

    Never raises; missing executables return ``(127, "", reason)``.
    """
    exe = cmd[0]
    if shutil.which(exe) is None:
        return 127, "", f"{exe} not found in PATH"
    try:
        proc = subprocess.run(
            list(cmd),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        return 124, "", f"{exe} timed out after {timeout}s: {e}"
    except OSError as e:
        return 126, "", f"{exe} failed: {e}"
    return proc.returncode, proc.stdout or "", proc.stderr or ""
