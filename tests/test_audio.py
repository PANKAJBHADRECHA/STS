import array
from pathlib import Path
import tempfile
import unittest
import wave

from sts_lab.audio import AudioBacklog, FrameQueue, inspect_wav, normalize_wav


class AudioTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)/"input.wav"

    def tearDown(self):
        self.tmp.cleanup()

    def make_wav(self, rate=24000, channels=1, samples=2400, amplitude=0):
        with wave.open(str(self.path), "wb") as stream:
            stream.setnchannels(channels)
            stream.setsampwidth(2)
            stream.setframerate(rate)
            stream.writeframes(array.array("h", [amplitude]*samples*channels).tobytes())

    def test_silence_and_clipping_are_measured(self):
        self.make_wav()
        self.assertEqual(inspect_wav(self.path)["rms"],0)
        self.make_wav(amplitude=32767)
        self.assertEqual(inspect_wav(self.path)["clipped_fraction"],1)

    def test_each_supported_rate_and_stereo_normalize(self):
        for rate in (16000,24000,44100,48000):
            for channels in (1,2):
                self.make_wav(rate,channels,rate//10,amplitude=1000)
                result=normalize_wav(self.path,Path(self.tmp.name)/"out.wav")
                self.assertEqual(result["sample_rate"],24000)
                self.assertEqual(result["channels"],1)
                self.assertAlmostEqual(result["duration_seconds"],.1,places=3)
                self.assertAlmostEqual(result["rms"],1000/32768)

    def test_empty_and_invalid_audio_rejected(self):
        self.make_wav(samples=0)
        with self.assertRaises(ValueError): inspect_wav(self.path)
        self.path.write_bytes(b"not a wav")
        with self.assertRaises(ValueError): inspect_wav(self.path)

    def test_truncated_file_rejected(self):
        self.make_wav()
        self.path.write_bytes(self.path.read_bytes()[:-10])
        with self.assertRaisesRegex(ValueError,"truncated"): inspect_wav(self.path)

    def test_unsupported_rate_rejected(self):
        self.make_wav(rate=8000)
        with self.assertRaises(ValueError): inspect_wav(self.path)

    def test_bounded_queue_stops_and_cleans_up(self):
        queue=FrameQueue()
        for _ in range(6): queue.put(bytes(3840))
        with self.assertRaises(AudioBacklog): queue.put(bytes(3840))
        self.assertTrue(queue.closed)
        self.assertIsNone(queue.get())
        with self.assertRaises(RuntimeError): queue.put(bytes(3840))

    def test_queue_rejects_partial_frames_and_preserves_order(self):
        queue=FrameQueue()
        with self.assertRaises(ValueError): queue.put(bytes(10))
        queue.put(bytes(3840));queue.put(bytes([1])*3840)
        self.assertEqual(queue.get(),bytes(3840))
        self.assertEqual(queue.get(),bytes([1])*3840)
