"""Loopback-only interactive estimator using the same torch-free Python core."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .estimator.memory import EstimateRequest, estimate
from .estimator.model_metadata import KNOWN_MODELS


class DemoInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    model_id: str
    method: str = "qlora"
    gpu_vram_gb: float = Field(16, gt=0, le=96)
    available_vram_gb: float | None = Field(None, gt=0, le=96)
    seq_len: int = Field(2048, ge=2, le=8192)
    micro_batch_size: int = Field(1, ge=1, le=8)
    lora_rank: int = Field(16, ge=1, le=128)
    optimizer: str = "paged_adamw_8bit"
    base_dtype: str = "bf16"
    gradient_checkpointing: bool = True
    attention_implementation: str = "sdpa"


def demo_estimate(payload):
    config = DemoInput.model_validate(payload)
    if config.model_id not in KNOWN_MODELS:
        raise ValueError(
            "demo supports the bundled model catalogue only; use the CLI for other model metadata"
        )
    if config.available_vram_gb is not None and config.available_vram_gb > config.gpu_vram_gb:
        raise ValueError("Currently free memory cannot exceed total VRAM.")
    request = EstimateRequest(**config.model_dump(), use_network=False)
    result = estimate(request).model_dump()
    parts = [
        "canifinetune estimate",
        "--model",
        config.model_id,
        "--method",
        config.method,
        "--gpu-vram-gb",
        str(config.gpu_vram_gb),
        "--seq-len",
        str(config.seq_len),
        "--micro-batch-size",
        str(config.micro_batch_size),
        "--lora-rank",
        str(config.lora_rank),
        "--optimizer",
        config.optimizer,
        "--base-dtype",
        config.base_dtype,
        "--attn",
        config.attention_implementation,
        "--offline",
    ]
    parts.append(
        "--gradient-checkpointing"
        if config.gradient_checkpointing
        else "--no-gradient-checkpointing"
    )
    if config.available_vram_gb is not None:
        parts.extend(["--available-vram-gb", str(config.available_vram_gb)])
    result["command"] = " ".join(parts)
    result["recipe_command"] = (
        result["command"]
        .replace("canifinetune estimate", "canifinetune recipe")
        .replace(" --offline", "")
        + " --output my-recipe"
    )
    # recipe shows a budget planning estimate using total memory. Free-memory
    # input is a time-local decision and is not persisted into future training.
    if config.available_vram_gb is not None:
        result["recipe_command"] = result["recipe_command"].replace(
            f" --available-vram-gb {config.available_vram_gb}", ""
        )
    return result


def make_server(port=8765):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send(self, status, data, content_type="application/json"):
            body = data if isinstance(data, bytes) else json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def valid_host(self):
            return self.headers.get("Host") in {
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            }

        def do_GET(self):  # noqa: N802
            if not self.valid_host():
                return self.send(403, {"error": "loopback Host required"})
            if self.path == "/api/models":
                return self.send(200, sorted(KNOWN_MODELS))
            files = {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/demo.js": ("demo.js", "text/javascript; charset=utf-8"),
                "/demo.css": ("demo.css", "text/css; charset=utf-8"),
            }
            if self.path not in files:
                return self.send(404, {"error": "not found"})
            filename, content_type = files[self.path]
            return self.send(
                200, (Path(__file__).parent / "web" / filename).read_bytes(), content_type
            )

        def do_POST(self):  # noqa: N802
            origin = self.headers.get("Origin")
            if not self.valid_host() or (
                origin
                and origin
                not in {
                    f"http://127.0.0.1:{self.server.server_port}",
                    f"http://localhost:{self.server.server_port}",
                }
            ):
                return self.send(403, {"error": "loopback origin required"})
            if self.path != "/api/estimate":
                return self.send(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if (
                    length <= 0
                    or length > 16384
                    or self.headers.get("Content-Type") != "application/json"
                ):
                    raise ValueError("send a JSON body of 1 to 16384 bytes")
                result = demo_estimate(json.loads(self.rfile.read(length)))
                return self.send(200, result)
            except ValidationError as exc:
                messages = [
                    error["msg"].removeprefix("Value error, ")
                    for error in exc.errors(include_url=False, include_input=False)
                ]
                return self.send(400, {"error": "; ".join(messages)})
            except (ValueError, TypeError) as exc:
                return self.send(400, {"error": str(exc)})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(port=8765):
    with make_server(port) as server:
        print(
            f"canifinetune demo: http://127.0.0.1:{server.server_port} (local, catalogue metadata only; Ctrl+C stops)",
            flush=True,
        )
        server.serve_forever()
