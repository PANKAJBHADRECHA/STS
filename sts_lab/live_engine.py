"""Local streaming ASR + MLX language model + streaming neural speech.

Capture/VAD/ASR stay active while response generation and playback proceed.
Every reply has an epoch; cancellation invalidates all of its future output.
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import queue
import re
import threading
import time

from .live_paths import ASSETS

SYSTEM = "You are a helpful voice assistant. Give brief, direct answers. User facts belong to the user. Follow the latest correction."


def complete_question(text):
    """Restore omitted terminal punctuation without changing recognized words."""
    if (
        text
        and text[-1] not in ".?!"
        and re.match(
            r"^(what|which|who|whose|where|when|why|how|is|are|am|do|does|did|can|could|would|will|should|have|has)\b",
            text,
            re.I,
        )
    ):
        return text + "?"
    return text


class LiveEngine:
    def __init__(
        self,
        sink,
        model="qwen-2b",
        voice="piper_en_US-lessac-medium",
        arch="MEDIUM_STREAMING",
        endpoint_ms=420,
    ):
        self.sink = sink
        self.diagnostics = False
        self.browser_audio = False
        self.model_name = model
        self.voice = voice
        self.arch_name = arch
        self.endpoint = endpoint_ms / 1000
        self.lock = threading.RLock()
        self.epoch = 0
        self.history = []
        self.summary = ""
        self.generated_current = ""
        self.reference_active_at = 0.0
        self.chunks = {}
        self.rendered = {}
        self.responding = False
        self.audio_queue = queue.Queue(maxsize=100)
        self.closed = threading.Event()
        self.reset_signal = threading.Event()
        self.responses = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="live-response"
        )
        self.metrics = {
            "audio_frames": 0,
            "input_overflows": 0,
            "asr_wall_s": 0.0,
            "input_s": 0.0,
            "interruptions": 0,
            "utterances": 0,
            "replies": 0,
            "errors": 0,
        }

    def emit(self, event, **data):
        self.sink({"type": event, **data})

    def load(self, emit_ready=True):
        import mlx.core as mx
        import moonshine_voice as mv
        from mlx_lm import load
        import webrtcvad

        mx.set_cache_limit(64 * 1024**2)
        mx.set_memory_limit(3 * 1024**3)
        self.model, self.tokenizer = load(str(ASSETS / self.model_name))
        arch = getattr(mv.ModelArch, self.arch_name)
        folder = {
            "SMALL_STREAMING": "small-streaming-en",
            "TINY_STREAMING": "tiny-streaming-en",
            "MEDIUM_STREAMING": "medium-streaming-en",
        }[self.arch_name]
        path = (
            ASSETS
            / "moonshine/download.moonshine.ai/model"
            / folder
            / "quantized_26_11_02"
        )
        if not path.is_dir():
            raise FileNotFoundError(
                "Run Setup Conversation.command to download local speech assets"
            )
        self.asr = mv.Transcriber(path, arch, update_interval=0.16)
        self.tts = (
            mv.TextToSpeech()
            .language("en-us")
            .voice(self.voice)
            .models_from(ASSETS / "moonshine", download=False)
            .load()
        )
        self.vad = webrtcvad.Vad(2)
        from .echo import EchoCanceller

        self.echo = EchoCanceller()
        self.tts.synthesize("Hello.")

        def warm_language_model():
            # Warm generation and neural voice before opening the microphone.
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler

            p = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": "Say hello."}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            generate(
                self.model,
                self.tokenizer,
                prompt=p,
                max_tokens=8,
                sampler=make_sampler(temp=0),
                verbose=False,
            )

        self.responses.submit(warm_language_model).result()
        self.asr_thread = threading.Thread(
            target=self._listen, name="live-asr", daemon=True
        )
        self.asr_thread.start()
        if emit_ready:
            self.emit(
                "ready",
                model=self.model_name,
                voice=self.voice,
                asr=self.arch_name,
                endpoint_ms=int(self.endpoint * 1000),
            )

    def feed(self, pcm):
        import numpy as np

        if self.closed.is_set():
            return
        a = np.frombuffer(pcm, dtype="<f4")
        if (
            len(a) not in (320, 640)
            or not np.isfinite(a).all()
            or np.max(np.abs(a)) > 1.01
        ):
            raise ValueError("Expected 20 ms finite mono float32 audio at 16 kHz")
        try:
            self.audio_queue.put_nowait(
                (
                    a[:320].copy(),
                    a[320:].copy()
                    if len(a) == 640
                    else np.zeros(320, dtype=np.float32),
                )
            )
        except queue.Full:
            self.metrics["input_overflows"] += 1
            raise

    def interrupt(self, reason="speech"):
        with self.lock:
            old = self.epoch
            was_active = self.responding
            self.epoch += 1
            if self.responding or self.chunks:
                self.metrics["interruptions"] += 1
                spoken = " ".join(
                    self.rendered.get(i, "") for i in sorted(self.rendered)
                ).strip()
                if spoken:
                    self.history.append(
                        {"role": "assistant", "content": spoken + " [interrupted]"}
                    )
                elif was_active:
                    # A conversation-control marker, never generated/unheard
                    # speech. Consecutive user turns confuse this small model.
                    self.history.append(
                        {
                            "role": "assistant",
                            "content": "[Response cancelled before completion.]",
                        }
                    )
            self.chunks.clear()
            self.rendered.clear()
            self.responding = False
            # Never wait on the synthesizer from the capture/ASR thread.
            # Epoch invalidation stops rendering immediately; the response
            # worker owns cancellation of its synthesis operation.
            self.emit(
                "cancel",
                epoch=old,
                next_epoch=self.epoch,
                reason=reason,
                at=time.monotonic(),
            )
            return self.epoch

    def ack(self, epoch, chunk, fraction=1.0):
        with self.lock:
            if epoch != self.epoch or chunk not in self.chunks:
                return
            text = self.chunks[chunk]
            fraction = max(0.0, min(1.0, float(fraction)))
            # A rendering estimate, never a claim that a human heard the words.
            if fraction >= 1 and text:
                self.rendered[chunk] = text

    def finish_playback(self, epoch):
        with self.lock:
            if epoch != self.epoch:
                return
            spoken = " ".join(
                self.rendered.get(i, "") for i in sorted(self.rendered)
            ).strip()
            if spoken:
                self.history.append({"role": "assistant", "content": spoken})
            self.chunks.clear()
            self.rendered.clear()
            self.responding = False
            self.emit("listening", epoch=epoch)

    def reset(self):
        self.reset_signal.set()
        self.interrupt("reset")
        with self.lock:
            self.history.clear()
            self.summary = ""
        self.emit("reset")

    def submit(self, text):
        text = text.strip()
        if (
            self.closed.is_set()
            or self.reset_signal.is_set()
            or not text
            or len(text) > 4000
        ):
            return
        text = complete_question(text)
        if text.lower().strip(" .!?") in ("stop", "wait", "cancel", "please stop"):
            epoch = self.interrupt("spoken control")
            with self.lock:
                self.history.extend(
                    [
                        {"role": "user", "content": text},
                        {
                            "role": "assistant",
                            "content": "[Speech stopped; waiting for the next request.]",
                        },
                    ]
                )
                self.metrics["utterances"] += 1
            self.emit("user", text=text, epoch=epoch)
            self.emit("listening", epoch=epoch)
            return
        epoch = self.interrupt("new utterance")
        with self.lock:
            # Bound a burst of typed requests while earlier inference cancels.
            self.history = self.history[-63:]
            self.history.append({"role": "user", "content": text})
            self.metrics["utterances"] += 1
            self.responding = True
        self.emit("user", text=text, epoch=epoch)
        previous = getattr(self, "pending_response", None)
        if previous:
            previous.cancel()
        self.pending_response = self.responses.submit(self._respond, text, epoch)

    def looks_like_echo(self, text):
        numbers = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
        }

        def norm(value):
            return " ".join(
                numbers.get(w, w) for w in re.findall(r"[a-z0-9]+", value.lower())
            )

        if text.lower().strip(" .!?") in ("stop", "wait", "cancel", "please stop"):
            return False
        heard = norm(text)
        spoken = norm(self.generated_current)
        return (
            time.monotonic() - self.reference_active_at < 2.0
            and bool(heard)
            and heard in spoken
        )

    def _listen(self):
        import numpy as np

        stream = None
        pre = deque(maxlen=40)
        voiced = 0
        silent = 0
        active = False
        latest = ""
        seconds = 0.0
        last_text_at = 0.0
        try:
            while not self.closed.is_set():
                if self.reset_signal.is_set():
                    if stream:
                        stream.close()
                        stream = None
                    pre.clear()
                    voiced = 0
                    silent = 0
                    active = False
                    latest = ""
                    seconds = 0.0
                    last_text_at = 0.0
                    while True:
                        try:
                            self.audio_queue.get_nowait()
                        except queue.Empty:
                            break
                    self.reset_signal.clear()
                try:
                    a, reference = self.audio_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                started = time.perf_counter()
                if (
                    self.browser_audio and self.responding and self.generated_current
                ) or np.sqrt(np.mean(reference * reference)) > 0.001:
                    self.reference_active_at = time.monotonic()
                raw_rms = float(np.sqrt(np.mean(a * a)))
                a = self.echo.process(a, reference)
                pre.append(a)
                speech = float(np.sqrt(np.mean(a * a))) > 0.002 and self.vad.is_speech(
                    (np.clip(a, -1, 1) * 32767).astype("<i2").tobytes(), 16000
                )
                voiced = voiced + 1 if speech else 0
                self.metrics["audio_frames"] += 1
                self.metrics["input_s"] += 0.02
                if self.metrics["audio_frames"] % 50 == 0:
                    self.emit(
                        "audio_stats",
                        raw_rms=raw_rms,
                        clean_rms=float(np.sqrt(np.mean(a * a))),
                        reference_rms=float(np.sqrt(np.mean(reference * reference))),
                        speech=speech,
                    )
                if not active and voiced >= 3:
                    # Suppress playback before potentially expensive ASR prefill.
                    self.emit("speech_start", at=time.monotonic())
                    confirmed = False
                    gain = min(
                        4.0,
                        max(
                            1.0,
                            0.08
                            / max(
                                0.001, max(float(np.sqrt(np.mean(f * f))) for f in pre)
                            ),
                        ),
                    )
                    if self.browser_audio:
                        gain = 1.0
                    stream = self.asr.create_stream(update_interval=0.16)
                    partial_lines = {}

                    def partial(event):
                        nonlocal confirmed, latest, last_text_at
                        if self.reset_signal.is_set():
                            return
                        if getattr(event, "line", None) is not None:
                            partial_lines[event.line.line_id] = event.line.text
                            value = " ".join(
                                partial_lines[k] for k in sorted(partial_lines)
                            ).strip()
                            if value and value != latest:
                                latest = value
                                last_text_at = time.monotonic()
                                if (
                                    not confirmed
                                    and (
                                        len(value.split()) >= 2
                                        or value.lower().strip(" .!?")
                                        in ("stop", "wait", "cancel", "no")
                                    )
                                    and not self.looks_like_echo(value)
                                ):
                                    confirmed = True
                                    self.interrupt("recognized speech")
                                if confirmed:
                                    self.emit("partial", text=value)

                    stream.add_listener(partial)
                    stream.start()
                    active = True
                    latest = ""
                    seconds = 0
                    silent = 0
                    for frame in pre:
                        stream.add_audio(np.clip(frame * gain, -1, 1).tolist(), 16000)
                elif active:
                    stream.add_audio(np.clip(a * gain, -1, 1).tolist(), 16000)
                if active:
                    seconds += 0.02
                    silent = 0 if speech else silent + 0.02
                    # Allow a longer pause after an unfinished clause. Never cut
                    # voiced speech solely because a partial has stopped changing.
                    unfinished = bool(
                        re.search(
                            r"\b(and|but|because|to|the|a|my|is|plus|actually)[,. ]*$",
                            latest.lower(),
                        )
                    )
                    endpoint = max(self.endpoint, 1.0) if unfinished else self.endpoint
                    if silent >= endpoint or seconds >= 25:
                        result = stream.stop()
                        text = " ".join(line.text for line in result.lines).strip()
                        stream.close()
                        stream = None
                        active = False
                        voiced = 0
                        pre.clear()
                        self.emit("speech_end", at=time.monotonic(), text=text)
                        # Avoid generating responses to quiet/noise-only segments.
                        if re.search(r"[A-Za-z0-9]", text) and not self.looks_like_echo(
                            text
                        ):
                            self.submit(text)
                        else:
                            self.emit("duck_restore")
                self.metrics["asr_wall_s"] += time.perf_counter() - started
        except Exception as error:
            self.metrics["errors"] += 1
            self.emit(
                "error", message="Audio recognition stopped: " + type(error).__name__
            )
        finally:
            if stream:
                try:
                    stream.close()
                except Exception:
                    pass

    def _respond(self, text, epoch):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        import mlx.core as mx
        import numpy as np

        def valid():
            return not self.closed.is_set() and self.epoch == epoch

        if not valid():
            return
        self.tts.next_chunk()  # Consume the previous reply's one-shot cancellation report.
        started = time.perf_counter()
        generated = []
        first_token = None
        first_audio = None
        try:
            # Compact older context using the same local model; commit only if
            # this generation still owns the session. Recent corrections stay raw.
            with self.lock:
                history = [dict(m) for m in self.history]
                summary = self.summary
            if len(history) > 32:
                source = json.dumps(history[:-16], ensure_ascii=False)[-12000:]
                prompt = self.tokenizer.apply_chat_template(
                    [
                        {
                            "role": "system",
                            "content": "Summarize conversation facts and latest corrections in under 120 words. Preserve names, dates, preferences. Do not add facts.",
                        },
                        {"role": "user", "content": summary + "\n" + source},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                compact = ""
                for r in stream_generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=180,
                    sampler=make_sampler(temp=0),
                ):
                    if not valid():
                        return
                    compact += r.text
                with self.lock:
                    if not valid():
                        return
                    self.summary = compact
                    self.history = self.history[-16:]
                    history = [dict(m) for m in self.history]
                    summary = compact
            system_text = SYSTEM + (
                "\nEarlier conversation summary: " + summary if summary else ""
            )
            messages = [{"role": "system", "content": system_text}] + history
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            # Keep worst-case typed input within this Mac's context budget.
            while len(prompt) > 3500 and len(history) > 1:
                history = history[2:] if len(history) > 2 else history[-1:]
                messages = [{"role": "system", "content": system_text}] + history
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            cached_tokens = 0
            self.generated_current = ""
            self.emit("reply_start", epoch=epoch)

            def tokens():
                nonlocal first_token
                for r in stream_generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=96,
                    sampler=make_sampler(temp=0),
                ):
                    if not valid():
                        return
                    if first_token is None:
                        first_token = time.perf_counter() - started
                    generated.append(r.text)
                    self.generated_current = "".join(generated)
                    yield r.text

            number = 0
            audio_seconds = 0.0
            phrase_text = {}
            voice_errors = []
            started_voice = threading.Event()

            def drain_voice():
                nonlocal first_audio, number, audio_seconds
                try:
                    while valid() and not started_voice.wait(0.02):
                        pass
                    while valid():
                        chunk = self.tts.next_chunk()
                        if chunk is None:
                            if not self.tts.is_streaming:
                                break
                            time.sleep(0.002)
                            continue
                        if not valid():
                            return
                        if first_audio is None:
                            first_audio = time.perf_counter() - started
                        samples = np.asarray(chunk.samples, dtype="<f4")
                        if not np.isfinite(samples).all():
                            raise ValueError("Nonfinite synthesized audio")
                        number += 1
                        audio_seconds += len(samples) / chunk.sample_rate
                        with self.lock:
                            if not valid():
                                return
                            if chunk.text:
                                phrase_text[chunk.utterance_id] = chunk.text
                            self.chunks[number] = (
                                phrase_text.get(chunk.utterance_id, "")
                                if chunk.is_final
                                else ""
                            )
                        self.sink(
                            {
                                "type": "audio",
                                "epoch": epoch,
                                "chunk": number,
                                "rate": chunk.sample_rate,
                                "text": chunk.text,
                                "samples": np.clip(samples, -1, 1).tobytes(),
                            }
                        )
                except Exception as error:
                    voice_errors.append(error)

            voice_thread = threading.Thread(
                target=drain_voice, name="live-synthesis", daemon=True
            )
            voice_thread.start()
            try:
                for piece in tokens():
                    if not valid():
                        break
                    self.tts.push_text(piece)
                    if piece:
                        started_voice.set()
                if valid():
                    self.tts.end_input()
                else:
                    self.tts.cancel_stream()
            finally:
                started_voice.set()
                voice_thread.join(timeout=15)
            if voice_thread.is_alive():
                raise TimeoutError("Speech worker did not finish")
            if voice_errors:
                raise voice_errors[0]
            if valid():
                self.metrics["replies"] += 1
                self.emit(
                    "reply_done",
                    epoch=epoch,
                    text="".join(generated),
                    first_token_s=first_token,
                    first_audio_s=first_audio,
                    audio_s=audio_seconds,
                    wall_s=time.perf_counter() - started,
                    peak_mlx_bytes=mx.get_peak_memory(),
                    cached_prompt_tokens=cached_tokens,
                )
        except Exception as error:
            if self.diagnostics:
                from .live_paths import LIVE_HOME
                import traceback

                diagnostics = LIVE_HOME / "diagnostics"
                diagnostics.mkdir(parents=True, exist_ok=True)
                with (diagnostics / "latest-response-error.log").open("w") as log:
                    traceback.print_exc(file=log)
            if valid():
                self.metrics["errors"] += 1
                self.emit("error", message="Response stopped: " + type(error).__name__)
        finally:
            if not valid():
                self.tts.cancel_stream()

    def close(self):
        if self.closed.is_set():
            return
        self.closed.set()
        self.interrupt("stop")
        if hasattr(self, "asr_thread"):
            self.asr_thread.join(timeout=5)
        self.responses.shutdown(wait=True, cancel_futures=True)
        if hasattr(self, "asr") and (
            not hasattr(self, "asr_thread") or not self.asr_thread.is_alive()
        ):
            self.asr.close()
        if hasattr(self, "echo"):
            self.echo.close()
        if hasattr(self, "tts"):
            self.tts.close()

        # Bound session lifetime even when WebSocket/audio callbacks form cycles.
        self.model = None
        self.tokenizer = None
        self.pending_response = None
        self.sink = lambda event: None
        try:
            import mlx.core as mx

            mx.clear_cache()
        except ImportError:
            pass
