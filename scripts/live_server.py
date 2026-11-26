"""Loopback-only full-duplex conversation service. No cloud inference."""

import argparse, asyncio, json, queue, sys, threading, subprocess, struct, os
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sts_lab.live_engine import LiveEngine


class Page(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.headers.get("Host") not in (
            f"127.0.0.1:{self.server.server_port}",
            f"localhost:{self.server.server_port}",
        ):
            self.send_error(403)
            return
        names = {
            "/": "live.html",
            "/live.js": "live.js",
            "/live.css": "live.css",
            "/live-worklet.js": "live-worklet.js",
        }
        name = names.get(self.path)
        if not name:
            self.send_error(404)
            return
        raw = (ROOT / "web" / name).read_bytes()
        self.send_response(200)
        self.send_header(
            "Content-Type",
            {"html": "text/html", "js": "text/javascript", "css": "text/css"}[
                name.rsplit(".", 1)[1]
            ]
            + "; charset=utf-8",
        )
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-STS-Application", "local-conversation-v1")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self' ws://127.0.0.1:* ws://localhost:*; frame-ancestors 'none'",
        )
        self.send_header("Permissions-Policy", "microphone=(self), camera=()")
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


async def main(args):
    from websockets.asyncio.server import serve

    active = asyncio.Lock()
    loop = asyncio.get_running_loop()

    async def session(ws):
        if active.locked():
            await ws.close(1013, "Another conversation is active")
            return
        async with active:
            awake = subprocess.Popen(
                ["caffeinate", "-i", "-w", str(__import__("os").getpid())]
            )
            out = asyncio.Queue(maxsize=64)
            closing = False
            audio = None

            def enqueue(event):
                if closing:
                    return
                try:
                    out.put_nowait(event)
                except asyncio.QueueFull:
                    asyncio.create_task(ws.close(1013, "Playback transport overloaded"))

            def sink(event):
                if closing:
                    return
                if args.diagnostics and event["type"] in (
                    "partial",
                    "user",
                    "cancel",
                    "speech_end",
                    "reply_done",
                    "error",
                ):
                    from sts_lab.live_paths import LIVE_HOME

                    trace = LIVE_HOME / "diagnostics/live-trace.jsonl"
                    trace.parent.mkdir(parents=True, exist_ok=True)
                    if trace.exists() and trace.stat().st_size > 256000:
                        trace.unlink()
                    with trace.open("a") as log:
                        log.write(json.dumps(event) + "\n")
                if event["type"] == "error":
                    print("Conversation error:", event.get("message"), flush=True)
                if audio:
                    audio.accept(event)
                    if event["type"] == "audio":
                        return
                if event["type"] == "audio":
                    # Backpressure synthesis, never capture, when the browser
                    # transport is full. No unbounded task/PCM accumulation.
                    asyncio.run_coroutine_threadsafe(out.put(event), loop).result(
                        timeout=10
                    )
                else:
                    loop.call_soon_threadsafe(enqueue, event)

            engine = LiveEngine(
                sink,
                model=args.model,
                voice=args.voice,
                arch=args.asr,
                endpoint_ms=args.endpoint_ms,
            )
            engine.diagnostics = args.diagnostics
            engine.browser_audio = args.audio == "browser"
            sender = None
            try:
                await ws.send(json.dumps({"type": "loading"}))

                async def send_events():
                    while True:
                        event = await out.get()
                        if event["type"] == "audio":
                            await ws.send(
                                struct.pack(
                                    "!III",
                                    event["epoch"],
                                    event["chunk"],
                                    event["rate"],
                                )
                                + event["samples"]
                            )
                        else:
                            await ws.send(json.dumps(event))

                sender = asyncio.create_task(send_events())
                await ws.send(json.dumps({"type": "audio_mode", "audio": args.audio}))
                if args.audio == "native":
                    from sts_lab.native_audio import NativeAudio

                    audio = NativeAudio(
                        engine, lambda e: loop.call_soon_threadsafe(enqueue, e)
                    )
                await asyncio.to_thread(engine.load, False)
                if audio:
                    await asyncio.to_thread(audio.start)
                mute = (
                    subprocess.check_output(
                        ["osascript", "-e", "output muted of (get volume settings)"],
                        text=True,
                    ).strip()
                    == "true"
                )
                sink(
                    {
                        "type": "ready",
                        "browser_audio": args.audio == "browser",
                        "model": args.model,
                        "voice": args.voice,
                        "asr": args.asr,
                        "devices": audio.devices if audio else None,
                        "speakers_muted": mute,
                    }
                )
                async for raw in ws:
                    if isinstance(raw, bytes):
                        if args.audio != "browser":
                            raise ValueError("Native audio input only")
                        deadline = loop.time() + 2.0
                        while True:
                            try:
                                engine.feed(raw)
                                break
                            except queue.Full:
                                if loop.time() >= deadline:
                                    raise
                                await asyncio.sleep(0.01)
                    else:
                        item = json.loads(raw)
                        if not isinstance(item, dict):
                            raise ValueError("Control message must be an object")
                        kind = item.get("type")
                        if kind == "interrupt":
                            engine.interrupt("manual")
                        elif kind == "ack" and args.audio == "browser":
                            engine.ack(int(item["epoch"]), int(item["chunk"]))
                        elif kind == "played" and args.audio == "browser":
                            engine.finish_playback(int(item["epoch"]))
                        elif kind == "text":
                            engine.submit(str(item.get("text", ""))[:4000])
                        elif kind == "reset":
                            engine.reset()
                        elif kind == "stop":
                            break
                        else:
                            raise ValueError("Unknown control message")
            except (ValueError, queue.Full, KeyError, TypeError) as error:
                print(
                    "Input/protocol error:",
                    type(error).__name__,
                    str(error),
                    flush=True,
                )
                await ws.close(1008, "Invalid audio or overloaded input; start again")
            except Exception as error:
                try:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "error",
                                "message": "Conversation stopped: "
                                + type(error).__name__
                                + ". Check local assets and macOS microphone permission; then start again.",
                            }
                        )
                    )
                except Exception:
                    pass
            finally:
                closing = True
                if audio:
                    await asyncio.to_thread(audio.close)
                while not out.empty():
                    out.get_nowait()
                if sender:
                    sender.cancel()
                    await asyncio.gather(sender, return_exceptions=True)
                await asyncio.to_thread(engine.close)
                awake.terminate()
                awake.wait(timeout=3)
                import gc

                gc.collect()
                try:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "stopped",
                                "metrics": engine.metrics,
                                "microphone_released": not audio
                                or not hasattr(audio, "stream")
                                or audio.stream.closed,
                            }
                        )
                    )
                except Exception:
                    pass

    try:
        from urllib.request import urlopen

        with urlopen(f"http://127.0.0.1:{args.port}", timeout=1) as existing:
            if existing.headers.get("X-STS-Application") == "local-conversation-v1":
                print("Local Conversation is already running", flush=True)
                if args.open:
                    import webbrowser

                    webbrowser.open(f"http://127.0.0.1:{args.port}")
                return
    except OSError:
        pass
    http = ThreadingHTTPServer(("127.0.0.1", args.port), Page)
    threading.Thread(target=http.serve_forever, daemon=True).start()
    if args.open:
        import webbrowser

        webbrowser.open(f"http://127.0.0.1:{args.port}")
    print(f"Local conversation: http://127.0.0.1:{args.port}", flush=True)
    try:
        async with serve(
            session,
            "127.0.0.1",
            args.port + 1,
            origins=[f"http://127.0.0.1:{args.port}", f"http://localhost:{args.port}"],
            max_size=8192,
            max_queue=50,
        ):
            await asyncio.Future()
    finally:
        http.shutdown()
        http.server_close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--open", action="store_true")
    p.add_argument("--audio", choices=["native", "browser"], default="browser")
    p.add_argument(
        "--diagnostics",
        action="store_true",
        help="Save a bounded private test transcript under Application Support",
    )
    p.add_argument(
        "--model", default="qwen-2b", choices=["qwen-08b", "qwen-2b", "lfm-text"]
    )
    p.add_argument(
        "--voice",
        default="piper_en_US-lessac-medium",
        choices=["kokoro_af_heart", "piper_en_US-lessac-medium"],
    )
    p.add_argument(
        "--asr",
        default="MEDIUM_STREAMING",
        choices=["TINY_STREAMING", "SMALL_STREAMING", "MEDIUM_STREAMING"],
    )
    p.add_argument("--endpoint-ms", type=int, default=420)
    asyncio.run(main(p.parse_args()))
