"""Switching voices while converting: load in the background, swap between chunks."""

import time
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from engine import core
from engine.settings import EngineSettings, save_model_settings
from tests.test_model_library import ModelFolders
from tools import cuda_graph


class FakeRVC:
    def __init__(self, tgt_sr=32000, if_f0=1):
        self.net_g = torch.nn.Linear(2, 2)
        self.tgt_sr, self.if_f0, self.version, self.is_half = tgt_sr, if_f0, "v2", False
        self.index_rate, self.pth_path, self.index_path = 0.0, "old.pth", ""
        self.key = self.formant = self.protect = None
        self.embedder, self.model = "contentvec", torch.nn.Linear(2, 2)
        self.embedders = {"contentvec": self.model}

    def change_key(self, key):
        self.key = key

    def change_formant(self, formant):
        self.formant = formant

    def change_protect(self, protect):
        self.protect = protect


class ModelSwapTest(ModelFolders):
    def running_engine(self, tgt_sr=32000):
        engine = self.engine(selected="Alto")
        save_model_settings(str(self.root / "Bass"), EngineSettings(pitch=7.5, output_gain_db=3.0))
        engine.running, engine.function = True, "vc"
        engine.rvc = FakeRVC()
        engine.config = SimpleNamespace(device=torch.device("cpu"))
        engine.input_wav_res = torch.zeros(160 * 600)
        engine.skip_head, engine.return_length = 500, 19
        engine.model_swap = engine.pending_swap = engine.swapped_model = None
        new_net = torch.nn.Linear(2, 2)
        cpt = {"config": [0] * 17 + [tgt_sr], "f0": 1, "version": "v2"}
        patches = [
            mock.patch.object(core.rvc_for_realtime, "get_synthesizer", return_value=(new_net, cpt)),
            mock.patch.object(core.rvc_for_realtime, "warm_synthesizer"),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        return engine, new_net

    def wait_for_load(self, engine):
        engine.model_swap["thread"].join(5)
        engine.tick = None  # not used: call the step directly
        engine.finish_model_swap()

    def test_switching_while_converting_keeps_the_stream_running(self):
        engine, new_net = self.running_engine()
        engine.update_settings({"model_name": "Bass"})
        self.assertTrue(engine.running)
        self.assertEqual(engine.settings.model_name, "Alto")  # old voice until the swap
        self.wait_for_load(engine)
        self.assertIsNotNone(engine.pending_swap)
        engine.apply_model_swap(engine.pending_swap)  # what the audio thread does
        self.assertIs(engine.rvc.net_g, new_net)
        self.assertTrue(new_net._rvc_cuda_graph_eager)
        self.assertEqual(engine.settings.model_name, "Bass")
        self.assertEqual((engine.settings.pitch, engine.settings.output_gain_db), (7.5, 3.0))
        self.assertEqual(engine.rvc.key, 7.5)
        engine.finish_model_swap()
        codes = [data["code"] for event, data in engine.events if event == "status"]
        self.assertEqual(codes[-2:], ["switching_model", "model_switched"])

    def test_a_different_sample_rate_restarts_instead(self):
        engine, _ = self.running_engine(tgt_sr=40000)
        engine.stop = mock.Mock(side_effect=lambda: setattr(engine, "running", False))
        engine.start = mock.Mock()
        engine.update_settings({"model_name": "Bass"})
        self.wait_for_load(engine)
        self.assertIsNone(engine.pending_swap)
        engine.stop.assert_called_once()
        engine.start.assert_called_once_with("vc")
        self.assertEqual(engine.settings.model_name, "Bass")

    def swap_to(self, engine, name, cpt):
        with mock.patch.object(
            core.rvc_for_realtime, "get_synthesizer", return_value=(torch.nn.Linear(2, 2), cpt)
        ):
            engine.update_settings({"model_name": name})
            self.wait_for_load(engine)

    def test_a_model_on_another_embedder_brings_it_along(self):
        engine, _ = self.running_engine()
        spin = torch.nn.Linear(2, 2)
        cpt = {"config": [0] * 17 + [32000], "f0": 1, "version": "v2", "embedder_model": "spin-v2"}
        with (
            mock.patch.object(core, "load_hubert_model", return_value=spin) as load,
            mock.patch.object(core, "warm_embedder") as warm,
        ):
            self.swap_to(engine, "Bass", cpt)
        load.assert_called_once_with(engine.config.device, False, "spin-v2")
        warm.assert_called_once()
        self.assertTrue(spin._rvc_cuda_graph_eager)  # no capture on the audio thread
        engine.apply_model_swap(engine.pending_swap)
        self.assertIs(engine.rvc.model, spin)
        self.assertEqual(engine.rvc.embedder, "spin-v2")

    def test_a_loaded_embedder_is_reused(self):
        engine, _ = self.running_engine()
        contentvec = engine.rvc.model
        cpt = {"config": [0] * 17 + [32000], "f0": 1, "version": "v2", "embedder_model": "contentvec"}
        with mock.patch.object(core, "load_hubert_model") as load:
            self.swap_to(engine, "Bass", cpt)
        load.assert_not_called()
        engine.apply_model_swap(engine.pending_swap)
        self.assertIs(engine.rvc.model, contentvec)

    def test_an_unsupported_embedder_keeps_the_current_voice(self):
        engine, _ = self.running_engine()
        cpt = {"config": [0] * 17 + [32000], "f0": 1, "version": "v2", "embedder_model": "other"}
        self.swap_to(engine, "Bass", cpt)
        self.assertIsNone(engine.pending_swap)
        self.assertEqual(engine.settings.model_name, "Alto")

    def test_stopping_while_loading_just_selects_the_model(self):
        engine, _ = self.running_engine()
        engine.update_settings({"model_name": "Bass"})
        engine.running = False
        self.wait_for_load(engine)
        self.assertIsNone(engine.pending_swap)
        self.assertEqual(engine.settings.model_name, "Bass")

    def test_a_newer_request_wins(self):
        engine, _ = self.running_engine()
        engine.update_settings({"model_name": "Bass"})
        first = engine.model_swap
        engine.update_settings({"model_name": "Tenor"})
        first["thread"].join(5)
        self.assertTrue(first["cancelled"])
        self.wait_for_load(engine)
        self.assertEqual(engine.pending_swap["name"], "Tenor")

    def test_not_running_changes_the_model_directly(self):
        engine = self.engine(selected="Alto")
        engine.update_settings({"model_name": "Bass"})
        self.assertEqual(engine.settings.model_name, "Bass")
        self.assertIsNone(getattr(engine, "model_swap", None))


class EagerOwnerTest(unittest.TestCase):
    def test_a_swapped_in_model_never_captures_a_graph(self):
        owner = torch.nn.Module()
        owner._rvc_cuda_graph_eager = True
        with mock.patch.object(cuda_graph, "cuda_graph_enabled", return_value=True):
            result = cuda_graph.run_cuda_graph(owner, "x", lambda value: value + 1, torch.ones(2))
        torch.testing.assert_close(result, torch.full((2,), 2.0))
        self.assertEqual(cuda_graph.get_cuda_graph_stats(owner)["captures"], 0)


if __name__ == "__main__":
    unittest.main()
