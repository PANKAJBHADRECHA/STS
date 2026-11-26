import unittest
import importlib.util
from sts_lab.live_engine import LiveEngine


class TTS:
    def cancel_stream(self):
        pass

    def close(self):
        pass


class LiveStateTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.engine = LiveEngine(self.events.append)
        self.engine.tts = TTS()

    def tearDown(self):
        self.engine.close()

    def test_interruption_keeps_completed_speech_only(self):
        e = self.engine
        e.epoch = 3
        e.responding = True
        e.chunks = {1: "The meeting is Friday.", 2: "Not rendered"}
        e.ack(3, 1)
        e.ack(3, 2, 0.25)
        e.interrupt()
        self.assertEqual(
            e.history,
            [{"role": "assistant", "content": "The meeting is Friday. [interrupted]"}],
        )
        self.assertEqual(e.epoch, 4)
        self.assertFalse(e.responding)

    def test_stale_playback_cannot_change_new_context(self):
        e = self.engine
        e.epoch = 1
        e.chunks = {1: "old"}
        e.interrupt()
        e.chunks = {1: "new"}
        e.ack(1, 1)
        e.finish_playback(1)
        self.assertEqual(e.rendered, {})
        self.assertEqual(e.history, [])

    def test_completed_playback_commits_context_once(self):
        e = self.engine
        e.epoch = 2
        e.chunks = {1: "Four."}
        e.ack(2, 1)
        e.finish_playback(2)
        e.finish_playback(2)
        self.assertEqual(e.history, [{"role": "assistant", "content": "Four."}])

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy required")
    def test_bad_audio_rejected_and_queue_is_bounded(self):
        import numpy as np
        import queue

        for a in [
            np.zeros(12, dtype="<f4"),
            np.full(320, np.nan, dtype="<f4"),
            np.full(320, 2, dtype="<f4"),
        ]:
            with self.assertRaises(ValueError):
                self.engine.feed(a.tobytes())
        for _ in range(100):
            self.engine.feed(np.zeros(640, dtype="<f4").tobytes())
        with self.assertRaises(queue.Full):
            self.engine.feed(np.zeros(640, dtype="<f4").tobytes())

    def test_reset_rejects_old_recognition_until_stream_is_flushed(self):
        e = self.engine
        e.history = [{"role": "user", "content": "Old fact"}]
        e.summary = "old summary"
        e.reset()
        epoch = e.epoch
        e.submit("Old final transcript racing reset")
        self.assertEqual(e.history, [])
        self.assertEqual(e.summary, "")
        self.assertEqual(e.epoch, epoch)
        self.assertTrue(e.reset_signal.is_set())

    def test_cancelled_unspoken_turn_has_control_marker_not_generated_text(self):
        e = self.engine
        e.history = [{"role": "user", "content": "Explain rain in six sentences."}]
        e.generated_current = "Unspoken generated answer"
        e.responding = True
        e.interrupt("recognized speech")
        self.assertEqual(
            e.history[-1],
            {"role": "assistant", "content": "[Response cancelled before completion.]"},
        )
        self.assertNotIn("Unspoken", e.history[-1]["content"])

    def test_recent_echo_does_not_cancel_but_explicit_stop_can(self):
        import time

        e = self.engine
        e.reference_active_at = time.monotonic() - 1.0
        e.generated_current = "Got it, Alex. I'm ready to help."
        self.assertTrue(e.looks_like_echo("Got it?"))
        self.assertFalse(e.looks_like_echo("Wait. What is three plus three?"))
        e.generated_current = "Stop."
        self.assertFalse(e.looks_like_echo("Stop."))

    def test_spoken_stop_is_a_control_not_a_new_generated_reply(self):
        e = self.engine
        e.responding = True
        e.submit("Please stop.")
        self.assertFalse(e.responding)
        self.assertEqual(e.metrics["replies"], 0)
        self.assertEqual(
            e.history[-1]["content"], "[Speech stopped; waiting for the next request.]"
        )

    def test_missing_question_mark_is_restored_without_changing_words(self):
        from sts_lab.live_engine import complete_question

        self.assertEqual(
            complete_question("What is four plus five"), "What is four plus five?"
        )
        self.assertEqual(complete_question("My name is Alex"), "My name is Alex")
        self.assertEqual(complete_question("What time?"), "What time?")

    def test_interrupt_never_waits_on_synthesis_from_recognition(self):
        class BusyVoice:
            def cancel_stream(self):
                raise AssertionError("Blocking synthesis called from interruption")

            def close(self):
                pass

        self.engine.tts = BusyVoice()
        self.engine.responding = True
        self.engine.interrupt("recognized speech")
        self.assertFalse(self.engine.responding)


class EchoTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy required")
    def test_known_delayed_echo_is_attenuated(self):
        import numpy as np
        from sts_lab.echo import EchoCanceller

        try:
            e = EchoCanceller()
        except RuntimeError:
            self.skipTest("SpeexDSP not installed")
        rng = np.random.default_rng(7)
        ref = rng.normal(0, 0.08, 16000 * 5).astype(np.float32)
        mic = np.pad(ref[:-960], (960, 0)) * 0.5
        out = np.concatenate(
            [
                e.process(mic[i : i + 320], ref[i : i + 320])
                for i in range(0, len(mic), 320)
            ]
        )
        erle = 10 * np.log10(
            np.mean(mic[-16000:] ** 2) / (np.mean(out[-16000:] ** 2) + 1e-12)
        )
        e.close()
        self.assertGreater(erle, 12)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy required")
    def test_quiet_user_speech_is_preserved_without_playback(self):
        import numpy as np
        from sts_lab.echo import EchoCanceller

        try:
            e = EchoCanceller()
        except RuntimeError:
            self.skipTest("SpeexDSP not installed")
        audio = np.random.default_rng(19).normal(0, 0.003, 320).astype(np.float32)
        self.assertTrue(
            np.array_equal(e.process(audio, np.zeros(320, dtype=np.float32)), audio)
        )
        e.close()


@unittest.skipUnless(importlib.util.find_spec("sounddevice"), "SoundDevice required")
class NativeRenderTests(unittest.TestCase):
    def test_cancel_removes_queued_speech_before_next_callback(self):
        import numpy as np
        from sts_lab.native_audio import NativeAudio

        class Engine:
            def feed(self, raw):
                self.frame = np.frombuffer(raw, dtype="<f4")

        engine = Engine()
        audio = NativeAudio(engine, lambda e: None)
        audio.accept({"type": "reply_start", "epoch": 2})
        audio.accept(
            {
                "type": "audio",
                "epoch": 2,
                "rate": 16000,
                "chunk": 1,
                "samples": np.ones(640, dtype="<f4").tobytes(),
            }
        )
        out = np.empty((320, 1), dtype=np.float32)
        audio._callback(np.zeros_like(out), out, 320, None, False)
        self.assertTrue(np.all(out == 1))
        self.assertEqual(len(engine.frame), 640)
        audio.accept({"type": "cancel", "epoch": 2, "next_epoch": 3})
        audio._callback(np.zeros_like(out), out, 320, None, False)
        self.assertTrue(np.all(out == 0))
        audio.accept(
            {
                "type": "audio",
                "epoch": 2,
                "rate": 16000,
                "chunk": 2,
                "samples": np.ones(320, dtype="<f4").tobytes(),
            }
        )
        self.assertFalse(audio.pending)

    def test_suppressed_speech_is_not_acknowledged_as_rendered(self):
        import numpy as np
        from sts_lab.native_audio import NativeAudio

        class Engine:
            def feed(self, raw):
                pass

        audio = NativeAudio(Engine(), lambda e: None)
        audio.accept({"type": "reply_start", "epoch": 1})
        audio.accept(
            {
                "type": "audio",
                "epoch": 1,
                "rate": 16000,
                "chunk": 1,
                "samples": np.ones(320, dtype="<f4").tobytes(),
            }
        )
        audio.accept({"type": "speech_start"})
        audio.gain = 0.0
        out = np.empty((320, 1), dtype=np.float32)
        audio._callback(np.zeros_like(out), out, 320, None, False)
        self.assertTrue(np.all(out == 0))
        queued = list(audio.actions.queue)
        self.assertFalse(any(e[0] == "ack" for e in queued))

    def test_short_final_chunk_notifies_start_before_end(self):
        import numpy as np
        from sts_lab.native_audio import NativeAudio

        class Engine:
            def feed(self, raw):
                pass

        audio = NativeAudio(Engine(), lambda e: None)
        audio.accept({"type": "reply_start", "epoch": 1})
        audio.accept(
            {
                "type": "audio",
                "epoch": 1,
                "rate": 16000,
                "chunk": 1,
                "samples": np.ones(160, dtype="<f4").tobytes(),
            }
        )
        audio.accept({"type": "reply_done", "epoch": 1})
        out = np.empty((320, 1), dtype=np.float32)
        audio._callback(np.zeros_like(out), out, 320, None, False)
        actions = list(audio.actions.queue)
        self.assertEqual(actions[0][0], "event")
        self.assertEqual(actions[0][1]["type"], "playback_start")
        self.assertEqual([a[0] for a in actions[1:]], ["ack", "played"])

    def test_old_response_events_cannot_rewind_render_epoch(self):
        from sts_lab.native_audio import NativeAudio

        audio = NativeAudio(object(), lambda e: None)
        audio.accept({"type": "cancel", "epoch": 1, "next_epoch": 2})
        audio.accept({"type": "reply_start", "epoch": 1})
        audio.accept({"type": "reply_done", "epoch": 1})
        self.assertEqual(audio.epoch, 2)
        self.assertFalse(audio.done)
        self.assertTrue(audio.actions.empty())

    def test_legitimate_thirty_second_reply_fits_bounded_output(self):
        import numpy as np
        from sts_lab.native_audio import NativeAudio

        audio = NativeAudio(object(), lambda e: None)
        audio.accept({"type": "reply_start", "epoch": 1})
        audio.accept(
            {
                "type": "audio",
                "epoch": 1,
                "rate": 16000,
                "chunk": 1,
                "samples": np.zeros(16000 * 30, dtype="<f4").tobytes(),
            }
        )
        self.assertEqual(len(audio.pending[0]["audio"]), 16000 * 30)
