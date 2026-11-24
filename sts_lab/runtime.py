"""Read-only feasibility checks. Never download weights as a hidden fallback."""
from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

PERSONAPLEX_REVISION = "7b29cfc4d9110f6b9d831b141e62f3d16f0bbe77"
MOSHI_REVISION = "e6a55d2722a65870ef52a6c9f6ecfc0e90f38362"


def probe_model_access(repo="nvidia/personaplex-7b-v1", filename="model.safetensors"):
    request = urllib.request.Request(f"https://huggingface.co/{repo}/resolve/main/{filename}", method="HEAD")
    token = os.getenv("HF_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return {"accessible": True, "http_status": response.status}
    except urllib.error.HTTPError as error:
        return {"accessible": False, "http_status": error.code,
                "reason": "model_access_required" if error.code in (401, 403) else "asset_unavailable"}
    except (OSError, TimeoutError):
        return {"accessible": False, "reason": "network_unavailable"}


def doctor(network=False):
    memory = None
    if platform.system() == "Darwin":
        memory = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
    packages = {name: importlib.util.find_spec(name) is not None for name in ("mlx", "moshi_mlx", "personaplex_mlx")}
    result = {
        "platform": platform.system(), "architecture": platform.machine(), "python": platform.python_version(),
        "memory_bytes": memory, "free_disk_bytes": shutil.disk_usage(Path.cwd()).free,
        "packages": packages, "hf_token_configured": bool(os.getenv("HF_TOKEN")),
        "capabilities": {"controller_simulation": True, "wav_validation": True,
                         "native_sts_inference_verified": False, "live_audio_qualified": False,
                         "native_tool_call_integration": False, "neural_self_listening": False},
        "runtime_revisions": {"personaplex_mlx": PERSONAPLEX_REVISION, "moshi": MOSHI_REVISION},
        "notes": ["Installed packages do not establish model availability or live performance.",
                  "PersonaPlex runtime may fall back from missing q4 weights to full precision; do not use that fallback on 8 GB.",
                  "Playback acknowledgements prove client rendering, not that a human heard or understood audio."],
    }
    if network:
        result["personaplex_access"] = probe_model_access()
    return result
