"""Check the built site's internal pages, fragments and inline image resources."""

import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.targets = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(values["id"])
        if tag in {"a", "link"} and values.get("href"):
            self.targets.append(values["href"])
        if tag in {"img", "script", "source"} and values.get("src"):
            self.targets.append(values["src"])
        if tag == "source" and values.get("srcset"):
            self.targets.extend(item.strip().split()[0] for item in values["srcset"].split(","))


def verify(site):
    root = Path(site).resolve()
    parsed = {}
    for page in root.rglob("*.html"):
        document = Links()
        document.feed(page.read_text(encoding="utf-8"))
        parsed[page.resolve()] = document
    assert parsed, "no HTML output"
    failures = []
    checked = 0
    for page, document in parsed.items():
        for link in document.targets:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            path = unquote(url.path)
            if not path:
                target = page
            elif path.startswith("/"):
                path = path.removeprefix("/can-i-finetune-this/")
                target = root / path.lstrip("/")
            else:
                target = page.parent / path
            target = target.resolve()
            if target.is_dir():
                target /= "index.html"
            checked += 1
            if not target.is_file():
                failures.append(f"{page.relative_to(root)}: missing {link}")
            elif (
                url.fragment
                and target in parsed
                and unquote(url.fragment) not in parsed[target].ids
            ):
                failures.append(f"{page.relative_to(root)}: missing fragment {link}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Verified {len(parsed)} HTML pages and {checked} internal page/asset/fragment links.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=Path, required=True)
    verify(parser.parse_args().site)
