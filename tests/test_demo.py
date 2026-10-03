import json
import threading
import urllib.error
import urllib.request

import pytest

from canifinetune.demo import demo_estimate, make_server
from canifinetune.estimator.memory import EstimateRequest, estimate


def test_demo_calls_same_offline_core():
    result = demo_estimate(
        {
            "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "gpu_vram_gb": 16,
            "available_vram_gb": 8,
            "seq_len": 512,
        }
    )
    expected = estimate(
        EstimateRequest(
            model_id="Qwen/Qwen2.5-1.5B-Instruct",
            gpu_vram_gb=16,
            available_vram_gb=8,
            seq_len=512,
            use_network=False,
        )
    )
    assert result["memory"] == expected.memory.model_dump()
    assert result["feasible"] == expected.feasible
    assert "--available-vram-gb 8.0" in result["command"]


@pytest.mark.parametrize(
    "payload",
    [
        {"model_id": ""},
        {"model_id": "http://internal/secret"},
        {"model_id": "<script>"},
        {"model_id": "Qwen/Qwen2.5-0.5B-Instruct", "seq_len": 0},
    ],
)
def test_invalid_demo_input(payload):
    with pytest.raises(ValueError):
        demo_estimate(payload)


def test_local_http_serves_resources_and_rejects_remote_origin():
    server = make_server(0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        for path in ("/", "/demo.js", "/demo.css", "/api/models"):
            with urllib.request.urlopen(base + path) as response:
                assert response.status == 200
                assert response.read()
        request = urllib.request.Request(
            base + "/api/estimate",
            data=json.dumps({"model_id": "Qwen/Qwen2.5-0.5B-Instruct"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request) as response:
            assert json.load(response)["memory"]["total_estimated_gb"] > 0
        request.add_header("Origin", "https://untrusted.example")
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        assert error.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
