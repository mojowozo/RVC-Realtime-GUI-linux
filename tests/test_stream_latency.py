import unittest

from engine.core import plausible_stream_latency


class StreamLatencyTest(unittest.TestCase):
    def test_keeps_a_real_latency(self):
        self.assertAlmostEqual(plausible_stream_latency(0.02), 0.02)

    def test_uses_the_output_side_of_a_duplex_pair(self):
        self.assertAlmostEqual(plausible_stream_latency((0.01, 0.03)), 0.03)

    def test_ignores_a_wrapped_jack_latency(self):
        # 2**32 - 960 frames at 48 kHz, as PortAudio reported for a PipeWire JACK node.
        self.assertEqual(plausible_stream_latency((2**32 - 960) / 48000), 0.0)

    def test_ignores_negative_and_non_finite(self):
        self.assertEqual(plausible_stream_latency(-0.01), 0.0)
        self.assertEqual(plausible_stream_latency(float("nan")), 0.0)
        self.assertEqual(plausible_stream_latency(None), 0.0)


if __name__ == "__main__":
    unittest.main()
