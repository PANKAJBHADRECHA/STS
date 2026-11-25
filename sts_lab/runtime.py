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


def saved_hf_token():
    """Use the standard local CLI credential without exposing it in diagnostics."""
    if os.getenv("HF_TOKEN"):
        return os.environ["HF_TOKEN"].strip()
    home = Path(os.getenv("HF_HOME", Path.home()/".cache/huggingface"))
    paths = [Path(os.getenv("HF_TOKEN_PATH", home/"token")), Path.home()/".huggingface/token"]
    for path in paths:
        try:
            if path.is_file():
                value = path.read_text().strip()
                if value:
                    return value
        except OSError:
            continue
    return None


def probe_model_access(repo="nvidia/personaplex-7b-v1", filename="model.safetensors"):
    request = urllib.request.Request(f"https://huggingface.co/{repo}/resolve/main/{filename}", method="HEAD")
    token = saved_hf_token()
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
    voice_verified = False
    voice_report = Path(__file__).resolve().parent.parent / "reports/voice-browser.json"
    if voice_report.exists():
        try:
            evidence = json.loads(voice_report.read_text())
            voice_verified = evidence.get("status") == "passed" and evidence.get("inference_mocked") is False
        except (OSError, ValueError):
            pass
    result = {
        "platform": platform.system(), "architecture": platform.machine(), "python": platform.python_version(),
        "memory_bytes": memory, "free_disk_bytes": shutil.disk_usage(Path.cwd()).free,
        "packages": packages, "hf_token_configured": bool(saved_hf_token()),
        "capabilities": {"controller_simulation": True, "wav_validation": True,
                         "native_sts_inference_verified": voice_verified, "live_audio_qualified": voice_verified,
                         "turn_based_voice_verified": voice_verified, "full_duplex_verified": False,
                         "native_tool_call_integration": False, "neural_self_listening": False},
        "runtime_revisions": {"personaplex_mlx": PERSONAPLEX_REVISION, "moshi": MOSHI_REVISION},
        "notes": ["Installed packages do not establish model availability or live performance.",
                  "PersonaPlex runtime may fall back from missing q4 weights to full precision; do not use that fallback on 8 GB.",
                  "Playback acknowledgements prove client rendering, not that a human heard or understood audio."],
    }
    if network:
        result["personaplex_access"] = probe_model_access()
    return result
