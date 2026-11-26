"""Download pinned local inference assets. This is the only network setup step."""

import argparse, json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sts_lab.live_paths import ASSETS

MODELS = {
    "qwen-2b": (
        "mlx-community/Qwen3.5-2B-4bit",
        "674aaa7240b91e8012fcad5d791b7dfe5ba90207",
    ),
    "qwen-08b": (
        "mlx-community/Qwen3.5-0.8B-4bit",
        "da28692b5f139cb0ec58a356b437486b7dac7462",
    ),
    "lfm-text": (
        "mlx-community/LFM2.5-1.2B-Instruct-4bit",
        "dee2f8a2786e6648bb644a7ca40652842490034b",
    ),
}


def main():
    from huggingface_hub import snapshot_download
    from moonshine_voice import get_model_for_language, ModelArch, TextToSpeech

    p = argparse.ArgumentParser()
    p.add_argument("--alternatives", action="store_true")
    args = p.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)
    required=100*1024**2 if (ASSETS/"qwen-2b/model.safetensors").exists() else 5*1024**3
    if shutil.disk_usage(ASSETS).free < required:
        raise SystemExit("At least 5 GiB free space required for setup")
    for name, (repo, revision) in MODELS.items():
        if name != "qwen-2b" and not args.alternatives:
            continue
        snapshot_download(
            repo,
            revision=revision,
            local_dir=ASSETS / name,
            allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"],
        )
    root = ASSETS / "moonshine"
    arches = [ModelArch.MEDIUM_STREAMING] + (
        [ModelArch.SMALL_STREAMING, ModelArch.TINY_STREAMING]
        if args.alternatives
        else []
    )
    for arch in arches:
        path, _ = get_model_for_language("en", arch, cache_root=root)
        if Path(path).name != "quantized_26_11_02":
            raise SystemExit(
                "Speech asset revision changed; expected quantized_26_11_02"
            )
    voices = ["piper_en_US-lessac-medium"] + (
        ["kokoro_af_heart"] if args.alternatives else []
    )
    for voice in voices:
        model = (
            TextToSpeech()
            .language("en-us")
            .voice(voice)
            .models_from(root, download=True)
            .load()
        )
        model.close()
    from ctypes.util import find_library

    if (
        not find_library("speexdsp")
        and not Path("/opt/homebrew/lib/libspeexdsp.dylib").exists()
    ):
        raise SystemExit("Run brew install speexdsp, then launch again")
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "assets.json").write_text(
        json.dumps(
            {
                "model": MODELS["qwen-2b"],
                "moonshine_sdk": "0.1.6",
                "asr": "MEDIUM_STREAMING",
                "voice": "piper_en_US-lessac-medium",
            },
            indent=2,
        )
        + "\n"
    )
    import sysconfig

    site = Path(sysconfig.get_paths()["purelib"]) / "sitecustomize.py"
    if not site.exists():
        site.write_text(
            "# STS: local Python cache\nimport sys\nfrom pathlib import Path\nif sys.pycache_prefix is None:sys.pycache_prefix=str(Path(__file__).resolve().parents[4]/'pycache')\n"
        )
    app = ASSETS.parent / "app"
    for relative in [
        "sts_lab/__init__.py",
        "sts_lab/live_paths.py",
        "sts_lab/live_engine.py",
        "sts_lab/echo.py",
        "sts_lab/native_audio.py",
        "scripts/live_server.py",
        "web/live.html",
        "web/live.css",
        "web/live.js",
        "web/live-worklet.js",
    ]:
        target = app / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    print(
        "Ready. Launch Start Conversation.command. All conversation inference is local."
    )


if __name__ == "__main__":
    main()
