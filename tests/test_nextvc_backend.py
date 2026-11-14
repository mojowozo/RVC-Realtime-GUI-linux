"""Drive the engine with a NextVC voice and check the audio path.

The streaming backend converts each block as it arrives, so the engine must
hand it exactly the input blocks, in order, and pass its output through
without the RVC context window or splice.  Skipped when the voice's project
or a CUDA device is not available.
"""

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from engine.core import RealtimeEngine
from engine.settings import EngineSettings
from infer import nextvc_backend
from tools.model_registry import discover_models
from tools.wav_recorder import WavRecorder

RATE = 48000
MODELS = Path(__file__).resolve().parent.parent / "models"


def find_voice():
    for model in discover_models(MODELS):
        if nextvc_backend.is_nextvc(model.model_path):
            spec = json.loads(model.model_path.read_text())
            if os.path.isdir(spec["project"]):
                return model
    return None


def nextvc_engine(model, block_time, device):
    engine = RealtimeEngine.__new__(RealtimeEngine)
    engine.settings = EngineSettings(block_time=block_time, extra_time=0.5, pitch=0)
    engine.config = SimpleNamespace(device=device)
    engine.function = "vc"
    engine.running = False
    engine.recorder = WavRecorder()
    engine.chunk_tap = None
    engine.monitor_device_index = None
    engine.output_queue = None
    engine.monitor_queue = None
    engine.last_input_meter_update = 0.0
    engine.last_output_meter_update = 0.0
    engine.emit = lambda event, data: None
    engine.load_model(model)
    engine.prepare_inference(RATE)
    return engine


@unittest.skipUnless(torch.cuda.is_available() and find_voice(), "needs CUDA and a NextVC voice")
class NextVCBackendTest(unittest.TestCase):
    def test_registry_lists_the_voice_without_an_index(self):
        model = find_voice()
        self.assertTrue(str(model.model_path).endswith(".nextvc.json"))
        self.assertIsNone(model.index_path)

    def test_block_is_rounded_to_whole_frames(self):
        device = torch.device(f"cuda:{torch.cuda.device_count() - 1}")
        engine = nextvc_engine(find_voice(), block_time=0.05, device=device)
        self.assertEqual(engine.block_frame % nextvc_backend.BLOCK_MULTIPLE, 0)

    def test_engine_output_is_the_backend_output(self):
        device = torch.device(f"cuda:{torch.cuda.device_count() - 1}")
        model = find_voice()
        engine = nextvc_engine(model, block_time=0.04, device=device)
        block = engine.block_frame
        time = np.arange(RATE * 2) / RATE
        # A voiced-like tone with harmonics and a slow level change.
        signal = sum(np.sin(2 * np.pi * 150 * k * time) / k for k in range(1, 9))
        signal = (0.05 * signal * (0.6 + 0.4 * np.sin(2 * np.pi * 1.5 * time))).astype(np.float32)

        outputs = []
        for start in range(0, signal.shape[0] - block + 1, block):
            outputs.append(engine.audio_callback(signal[start : start + block, None].copy(), block, None, None).copy())
        output = np.concatenate(outputs)

        self.assertTrue(np.isfinite(output).all())
        self.assertGreater(float(np.abs(output[RATE:]).max()), 1e-3)
        # Same blocks straight through a second backend: identical up to the
        # random noise in the model's excitation, so compare the envelopes.
        direct = nextvc_backend.NextVC(0, str(model.model_path), device)
        reference = torch.cat(
            [
                direct.process(torch.from_numpy(signal[start : start + block]).to(device))
                for start in range(0, signal.shape[0] - block + 1, block)
            ]
        ).cpu().numpy()
        frame = RATE // 100
        envelope = lambda x: np.sqrt((x[: len(x) // frame * frame].reshape(-1, frame) ** 2).mean(1))  # noqa: E731
        got, want = envelope(output[RATE // 2 :]), envelope(reference[RATE // 2 :])
        self.assertGreater(float(np.corrcoef(got, want)[0, 1]), 0.98)


if __name__ == "__main__":
    unittest.main()
