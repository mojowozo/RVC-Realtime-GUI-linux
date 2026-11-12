"""RefineGAN decoder and per-model embedders, without any model weights."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import torch
import torchaudio

from infer import hubert, rtrvc
from infer.module.models import GeneratorNSF, SynthesizerTrnMs768NSFsid
from infer.module.refinegan import RefineGANGenerator, SineGenerator, sinc_downsampler

UPP = 16
SR = 16000
CONFIG = [
    513, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5]] * 3,
    [10, 8, 2, 2], 512, [20, 16, 4, 4], 1, 256, 32000,
]


def generator():
    return RefineGANGenerator(
        sample_rate=SR, upsample_rates=(2, 2, 2, 2), num_mels=8, gin_channels=256
    ).eval()


class SineGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.gen = SineGenerator(SR)
        self.gen.merge[0].weight.data.fill_(1.0)
        self.f0 = torch.full((1, 4000, 1), 220.0)

    def render(self, f0, phase0=None):
        phase = None if phase0 is None else torch.tensor([phase0])
        with torch.no_grad():
            return self.gen(f0, noise=torch.zeros_like(f0), phase0=phase)

    def overlap(self, shift, phase0):
        first = self.render(self.f0)[0, shift:]
        second = self.render(self.f0[:, shift:], phase0)[0, : first.shape[0]]
        return first, second

    def test_carried_phase_makes_overlapping_renders_agree(self):
        shift = 1234
        first, second = self.overlap(shift, float(220.0 * shift / SR % 1.0))
        self.assertLess((first - second).abs().max().item(), 1e-3)

    def test_without_carried_phase_they_disagree(self):
        first, second = self.overlap(1234, 0.0)
        self.assertGreater((first - second).abs().max().item(), 0.01)

    def test_explicit_noise_is_used_on_unvoiced_samples(self):
        f0 = torch.zeros(1, 100, 1)
        with torch.no_grad():
            silent = self.gen(f0, noise=torch.zeros_like(f0))
            noisy = self.gen(f0, noise=torch.ones_like(f0))
        self.assertTrue(torch.all(silent == 0))
        self.assertTrue(torch.all(noisy != 0))


class RefineGANGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.gen = generator()
        self.latent = torch.randn(1, 8, 12)
        self.f0 = torch.full((1, 12), 200.0)
        self.speaker = torch.zeros(1, 256, 1)

    def test_output_is_one_upsampled_frame_per_latent_frame(self):
        with torch.no_grad():
            audio = self.gen(self.latent, self.f0, g=self.speaker)
        self.assertEqual(audio.shape, (1, 1, 12 * UPP))

    def test_accepts_the_streaming_inputs(self):
        with torch.no_grad():
            audio = self.gen(
                self.latent, self.f0, g=self.speaker,
                noise=torch.zeros(12 * UPP), phase0=torch.tensor([0.25]),
            )
        self.assertEqual(audio.shape[-1], 12 * UPP)

    def test_n_res_stretches_for_formant_shifting(self):
        with torch.no_grad():
            audio = self.gen(self.latent, self.f0, g=self.speaker, n_res=15)
        self.assertEqual(audio.shape[-1], 15 * UPP)

    def test_filters_are_not_part_of_a_checkpoint(self):
        self.assertFalse([key for key in self.gen.state_dict() if "kernel" in key])

    def test_filters_match_the_functional_resample(self):
        signal = torch.randn(1, 2, 400)
        expected = torchaudio.functional.resample(
            signal, 8, 1, lowpass_filter_width=64, rolloff=0.9475937167399596,
            resampling_method="sinc_interp_kaiser", beta=14.769656459379492,
        )
        self.assertTrue(torch.allclose(sinc_downsampler(8, 1)(signal), expected, atol=1e-5))

    def test_weight_norm_removal_keeps_the_output(self):
        with torch.no_grad(), mock.patch("torch.randn_like", torch.zeros_like):
            noise = torch.zeros(12 * UPP)
            before = self.gen(self.latent, self.f0, g=self.speaker, noise=noise)
            self.gen.remove_weight_norm()
            after = self.gen(self.latent, self.f0, g=self.speaker, noise=noise)
        self.assertTrue(torch.allclose(before, after))


class CheckpointTest(unittest.TestCase):
    def test_the_synthesizer_builds_the_checkpoints_decoder(self):
        refine = SynthesizerTrnMs768NSFsid(*CONFIG, is_half=False, vocoder="RefineGAN")
        self.assertIsInstance(refine.dec, RefineGANGenerator)
        self.assertEqual(refine.dec.upp, 320)
        plain = SynthesizerTrnMs768NSFsid(*CONFIG, is_half=False)
        self.assertIsInstance(plain.dec, GeneratorNSF)

    def load(self, weights, **fields):
        cpt = {"config": list(CONFIG), "weight": weights, "f0": 1, "version": "v2", **fields}
        with mock.patch.object(rtrvc.torch, "load", return_value=cpt):
            return rtrvc.get_synthesizer("model.pth")

    def test_an_unknown_vocoder_is_refused(self):
        weights = {"emb_g.weight": torch.zeros(1, 256)}
        with self.assertRaisesRegex(ValueError, "MRF HiFi-GAN"):
            self.load(weights, vocoder="MRF HiFi-GAN")

    def test_a_decoder_with_missing_weights_is_refused(self):
        weights = SynthesizerTrnMs768NSFsid(*CONFIG, is_half=False, vocoder="RefineGAN").state_dict()
        del weights["dec.conv_post.weight_g"]
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.load(weights, vocoder="RefineGAN")

    def test_refinegan_weights_load(self):
        weights = SynthesizerTrnMs768NSFsid(*CONFIG, is_half=False, vocoder="RefineGAN").state_dict()
        net_g, _ = self.load(weights, vocoder="RefineGAN")
        self.assertIsInstance(net_g.dec, RefineGANGenerator)


class EmbedderTest(unittest.TestCase):
    def test_older_checkpoints_use_contentvec(self):
        self.assertEqual(hubert.checkpoint_embedder({}), "contentvec")
        self.assertEqual(hubert.checkpoint_embedder({"embedder_model": None}), "contentvec")

    def test_the_checkpoints_embedder_is_used(self):
        self.assertEqual(hubert.checkpoint_embedder({"embedder_model": "spin-v2"}), "spin-v2")

    def test_an_unknown_embedder_is_refused(self):
        with self.assertRaisesRegex(ValueError, "custom"):
            hubert.checkpoint_embedder({"embedder_model": "custom"})

    def test_missing_files_are_reported(self):
        with TemporaryDirectory() as folder:
            paths = {"spin-v2": Path(folder)}
            with mock.patch.object(hubert, "EMBEDDER_PATHS", paths):
                with self.assertRaises(FileNotFoundError):
                    hubert.load_hubert_model("cpu", False, "spin-v2")


if __name__ == "__main__":
    unittest.main()
