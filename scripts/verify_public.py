"""Verify exact official PyPI distribution identities against the tested candidate."""

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def verify(version, candidate):
    candidate = Path(candidate)
    expected = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in candidate.iterdir()
        if path.suffix in {".whl", ".gz"}
    }
    assert len(expected) == 2
    url = f"https://pypi.org/pypi/canifinetune/{version}/json"
    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                release = json.load(response)
            break
        except urllib.error.HTTPError as exc:
            if exc.code != 404 or attempt == 7:
                raise
            time.sleep(5)
    assert release["info"]["version"] == version
    actual = {file["filename"]: file["digests"]["sha256"] for file in release["urls"]}
    assert actual == expected, {"public": actual, "candidate": expected}
    for file in release["urls"]:
        assert file["url"].startswith("https://files.pythonhosted.org/")
        with urllib.request.urlopen(file["url"], timeout=30) as response:
            downloaded = response.read()
        assert hashlib.sha256(downloaded).hexdigest() == expected[file["filename"]]
    print(json.dumps({"public_version": version, "index": url, "sha256": actual}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    options = parser.parse_args()
    verify(options.version, options.candidate)
