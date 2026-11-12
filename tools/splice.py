"""WSOLA splicing of consecutive real-time inference chunks.

Each inference returns ``block + FADE + search`` samples (search = 10 ms).
The head of each chunk overlaps the previous chunk's *natural continuation*:
the samples it produced right after its emitted block.  WSOLA picks the
offset in ``[0, search]`` whose waveform best matches that continuation,
crossfades over a fixed 40 ms, and emits ``block`` samples.

The last ``search - offset`` samples of every chunk are never used.  They are
the model's causal edge, where it has no future context and edge artifacts
live, so splicing early in the window (small offsets) keeps the continuation
template clear of them.  Compared with the previous SOLA splice this

- treats the previous seam's offset as the expected position: a nearby,
  almost-as-good match is preferred over a distant one, and
- keeps the previous offset when no candidate is a confident match (silence,
  noise, unvoiced consonants), instead of jumping to a random argmax that can
  drag the template into the chunk's edge;
- reports a real confidence (correlation normalised by both energies).

Crossfading two renderings adds their waves, so wherever they differ the level
falls in the middle of the fade: up to 3 dB where they are unrelated, which is
most often in the highs.  At the block rate that is an audible seam, so the
fade measures how well the two match, separately below and above
``BAND_SPLIT_HZ``, and lifts each band by what it would lose.
"""

import numpy as np
import torch
import torch.nn.functional as F

FADE_SECONDS = 0.040
SEARCH_SECONDS = 0.010
#: Below this normalised correlation the seam is not a meaningful match.
#: Tuned on real RVC output (tools/splice_bench.py): 0.3 discarded usable
#: weak matches; 0.15 keeps them while still ignoring unmatched noise.
CONFIDENCE_THRESHOLD = 0.15
#: Score penalty for moving the splice by the whole search span.  0.05 cut
#: the time skipped or repeated at seams by about 20% with no loss of seam
#: confidence or spectral continuity on real RVC output.
CONTINUITY_PENALTY = 0.05
#: Mean template power below this (about -80 dBFS) counts as silence.
SILENCE_POWER = 1e-8
#: Two bands did as well as four or eight.
BAND_SPLIT_HZ = 4000.0
#: Largest lift; 1/sqrt(0.5) is enough for unrelated chunks of equal level.
MAX_FADE_GAIN = 1.5
#: Length of the band-splitting filter.
BAND_FILTER_SECONDS = 0.0053


def lowpass_filter(samplerate, device, dtype):
    """Windowed-sinc low-pass at ``BAND_SPLIT_HZ``, ready for ``F.conv1d``."""
    half = int(round(BAND_FILTER_SECONDS * samplerate / 2))
    n = torch.arange(-half, half + 1, device=device, dtype=torch.float64)
    cutoff = 2.0 * BAND_SPLIT_HZ / samplerate
    window = torch.hann_window(2 * half + 3, periodic=False, device=device, dtype=torch.float64)[1:-1]
    taps = cutoff * torch.sinc(cutoff * n) * window
    return (taps / taps.sum()).to(dtype)[None, None, :]


class WsolaSplicer:
    #: Hold the level through the fade (see the module docstring); the engine's
    #: ``matched_fade`` setting, live.  On 200 s of speech two voices had not been
    #: tuned on, at 0.05 and 0.10 s blocks, this took the block-rate seam score
    #: from 8.7-11.8 to 0.2-2.7 (a clear seam is above 3) and halved the level
    #: ripple over a block or better, with seam-free control periods unchanged.
    matched_fade = True

    def __init__(self, samplerate, block, device, dtype=torch.float32):
        zc = samplerate // 100
        # Read at construction so benchmarks can try other lengths.
        self.fade = int(round(FADE_SECONDS * 100)) * zc
        # One 10 ms frame, so the inference length stays a whole number of
        # frames at any rate (zc is odd at 44.1 kHz).
        self.search = zc
        self.block = block
        self.device = device
        ramp = torch.linspace(0.0, 1.0, steps=self.fade, device=device, dtype=dtype)
        self.fade_in = torch.sin(0.5 * np.pi * ramp) ** 2
        self.fade_out = 1 - self.fade_in
        self.ones = torch.ones(1, 1, self.fade, device=device, dtype=dtype)
        self.candidates = torch.arange(self.search + 1, device=device, dtype=dtype)
        self.template = torch.zeros(self.fade, device=device, dtype=dtype)
        self.last_offset = 0
        self.last_confidence = 0.0
        self.band_filter = lowpass_filter(samplerate, device, dtype)
        self.analysis_window = torch.hann_window(self.fade, device=device, dtype=dtype)
        # A Hann window's power, and rfft's scale, turn band energy into mean power.
        self.power_scale = 2.0 / (self.fade * torch.sum(self.analysis_window**2))
        frequencies = torch.fft.rfftfreq(self.fade, 1.0 / samplerate, device=device)
        self.low_bins = frequencies < BAND_SPLIT_HZ

    @property
    def input_length(self):
        """Samples each inference must return."""
        return self.block + self.fade + self.search

    def warm(self):
        """Do the GPU's one-time FFT setup (about 150 ms) before audio starts."""
        self.crossfade(torch.zeros_like(self.template))

    def reset(self):
        self.template.zero_()
        self.last_offset = 0
        self.last_confidence = 0.0

    def find_offset(self, infer_wav):
        """Return ``(offset, confidence)`` for splicing ``infer_wav``."""
        window = infer_wav[None, None, : self.fade + self.search]
        numerator = F.conv1d(window, self.template[None, None, :])[0, 0]
        candidate_energy = F.conv1d(window**2, self.ones)[0, 0]
        template_energy = torch.sum(self.template**2)
        ncc = numerator / torch.sqrt(candidate_energy * template_energy + 1e-12)
        penalty = CONTINUITY_PENALTY * (self.candidates - self.last_offset).abs() / self.search
        best = torch.argmax(ncc - penalty)
        best_offset, confidence, template_power = torch.stack(
            (best.to(ncc.dtype), ncc[best], template_energy / self.fade)
        ).tolist()
        if template_power < SILENCE_POWER or confidence < CONFIDENCE_THRESHOLD:
            return self.last_offset, confidence
        return int(best_offset), confidence

    def fade_gains(self, head):
        """Per-band gains over the fade; ``[low, high]``, 1.0 where nothing is lost."""
        previous = torch.fft.rfft(self.template * self.analysis_window)
        current = torch.fft.rfft(head * self.analysis_window)
        gains = []
        for bins in (self.low_bins, ~self.low_bins):
            a, b = previous[bins], current[bins]
            power_a = torch.sum(a.abs() ** 2) * self.power_scale
            power_b = torch.sum(b.abs() ** 2) * self.power_scale
            root = torch.sqrt(power_a * power_b)
            match = torch.sum((a * b.conj()).real) * self.power_scale / (root + 1e-12)
            match = match.clamp(0.0, 1.0)
            # The level a perfect match would give, so that only mismatch is
            # lifted: a word starting or ending is meant to change level.
            matched_level = self.fade_out * power_a.sqrt() + self.fade_in * power_b.sqrt()
            faded_power = (
                self.fade_out**2 * power_a
                + self.fade_in**2 * power_b
                + 2.0 * self.fade_out * self.fade_in * match * root
            )
            gain = matched_level / (faded_power.sqrt() + 1e-12)
            # Silence on either side has nothing to match.
            if torch.minimum(power_a, power_b) < SILENCE_POWER:
                gain = torch.ones_like(gain)
            gains.append(gain.clamp(1.0, MAX_FADE_GAIN))
        return gains

    def crossfade(self, head):
        """Fade from the template to ``head``, holding each band's level."""
        faded = head * self.fade_in + self.template * self.fade_out
        if not self.matched_fade:
            return faded
        low_gain, high_gain = self.fade_gains(head)
        padding = self.band_filter.shape[-1] // 2
        padded = F.pad(faded[None, None, :], (padding, padding), mode="reflect")
        low = F.conv1d(padded, self.band_filter)[0, 0]
        # Both gains are 1 at the fade's ends, so it still joins seamlessly.
        return low * low_gain + (faded - low) * high_gain

    def splice(self, infer_wav, crossfade=True):
        """Crossfade ``infer_wav`` onto the previous chunk; return ``block`` samples.

        ``crossfade=False`` starts fresh after silence (no previous chunk to
        join), so the head of the chunk is not faded in over 40 ms.
        """
        if not crossfade:
            self.template.zero_()
            self.last_offset, self.last_confidence = 0, 0.0
            output = infer_wav
            self.template[:] = output[self.block : self.block + self.fade]
            return output[: self.block]
        offset, confidence = self.find_offset(infer_wav)
        self.last_offset, self.last_confidence = offset, confidence
        output = infer_wav[offset:]
        # The samples right after the emitted block are the natural
        # continuation the next chunk is matched against.  Copy them before
        # fading: with a block shorter than the fade the regions overlap.
        continuation = output[self.block : self.block + self.fade].clone()
        output[: self.fade] = self.crossfade(output[: self.fade].clone())
        self.template[:] = continuation
        return output[: self.block]
