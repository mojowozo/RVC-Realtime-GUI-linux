"""Backend for the causal streaming voice model developed in the VoiceChanger
project (causal content encoder, pitch tracker and decoder).

Unlike RVC it converts each block as it arrives and its output is already
continuous across blocks, so the engine skips the context window, the pitch
cache and the WSOLA splice for it.  A voice is a folder in ``models/`` holding
``<anything>.nextvc.json``:

    {"project": "/path/to/VoiceChanger",
     "decoder": "runs/models/voice-NAME/decoder.pt",
     "pitch":   "runs/models/pitch-la2/pitch.pt",
     "speaker": "NAME"}

Relative paths are resolved against ``project``.  ``encoder`` is optional (the
decoder checkpoint names the one it was trained with).
"""

import json
import os
import sys

import torch

SUFFIX = ".nextvc.json"
#: Blocks must be a whole number of 20 ms encoder frames at 48 kHz.
SAMPLE_RATE = 48000
BLOCK_MULTIPLE = 960


def is_nextvc(model_path):
    return str(model_path).endswith(SUFFIX)


class NextVC:
    is_nextvc = True
    tgt_sr = SAMPLE_RATE
    # Attributes the engine looks for on an RVC model.
    net_g = None
    model = None
    input_f0 = None
    pitch_smoothing = 0.0

    def __init__(self, key, model_path, device):
        with open(model_path, encoding="utf-8") as handle:
            spec = json.load(handle)
        project = spec["project"]
        if project not in sys.path:
            sys.path.insert(0, project)
        from nextvc.stream import DELAY, StreamingVC

        def resolve(path):
            return path if path is None or os.path.isabs(path) else os.path.join(project, path)

        self.speaker = spec["speaker"]
        self.vc = StreamingVC(
            resolve(spec["decoder"]), resolve(spec["pitch"]), str(device), resolve(spec.get("encoder"))
        )
        if self.speaker not in self.vc.speakers:
            raise ValueError(f"{self.speaker} is not a voice of {spec['decoder']}")
        self.delay_seconds = getattr(self.vc, "delay", DELAY) / 100.0
        self.change_key(key)

    def change_key(self, key):
        self.vc.set_voice(self.speaker, float(key))

    def change_formant(self, formant):
        pass

    def change_index_rate(self, index_rate):
        pass

    def change_protect(self, protect):
        pass

    def reset_stream_state(self):
        self.vc.reset()

    def process(self, block):
        """One block of 48 kHz mono samples (a torch tensor) -> the same length."""
        return self.vc.process(block)

    def warm(self, block_frame):
        for _ in range(4):
            self.vc.process(torch.zeros(block_frame, device=self.vc.device))
        self.vc.reset()
