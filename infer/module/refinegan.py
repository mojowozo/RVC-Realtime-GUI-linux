"""RefineGAN decoder for RVC models trained with ``vocoder: "RefineGAN"``.

Ported from Applio (rvc/lib/algorithm/generators/refinegan.py, MIT License,
Copyright (c) AI Hispano).  The layers match Applio's, so its checkpoints load
unchanged.  The differences are all for streaming:

- ``SineGenerator`` accepts the time-aligned noise and starting phase that
  ``infer/rtrvc.py`` carries between chunks, as ``SineGen`` does for HiFi-GAN.
- The sinc filters are built once instead of on every call.
- ``forward`` takes the same arguments as ``GeneratorNSF.forward``.
"""

import numpy as np
import torch
import torchaudio
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

from infer.module.commons import get_padding, init_weights


def sinc_downsampler(orig_freq, new_freq):
    """The anti-aliased resampler between two of the sine's scales."""
    resampler = torchaudio.transforms.Resample(
        orig_freq=orig_freq,
        new_freq=new_freq,
        resampling_method="sinc_interp_kaiser",
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        beta=14.769656459379492,
    )
    # Derived from the arguments, so not part of a checkpoint.
    kernel = resampler._buffers.pop("kernel")
    resampler.register_buffer("kernel", kernel, persistent=False)
    return resampler


class ResBlock(nn.Module):
    """Dilated convolutions at several scales, with residual connections."""

    def __init__(self, channels, kernel_size=7, dilation=(1, 3, 5), leaky_relu_slope=0.2):
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope
        self.convs1 = nn.ModuleList(
            weight_norm(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=d,
                    padding=get_padding(kernel_size, d),
                )
            )
            for d in dilation
        )
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList(
            weight_norm(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                )
            )
            for _ in dilation
        )
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, self.leaky_relu_slope)
            xt = c1(xt)
            xt = F.leaky_relu(xt, self.leaky_relu_slope)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for c1, c2 in zip(self.convs1, self.convs2):
            remove_weight_norm(c1)
            remove_weight_norm(c2)


class AdaIN(nn.Module):
    """Add learned per-channel noise, then a leaky ReLU."""

    def __init__(self, *, channels, leaky_relu_slope=0.2):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels) * 1e-4)
        self.activation = nn.LeakyReLU(leaky_relu_slope)

    def forward(self, x):
        gaussian = torch.randn_like(x) * self.weight[None, :, None]
        return self.activation(x + gaussian)


class ParallelResBlock(nn.Module):
    """Residual blocks with different kernel sizes, averaged."""

    def __init__(
        self,
        *,
        in_channels,
        out_channels,
        kernel_sizes=(3, 7, 11),
        dilation=(1, 3, 5),
        leaky_relu_slope=0.2,
    ):
        super().__init__()
        self.input_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=7,
            stride=1,
            padding=3,
        )
        self.input_conv.apply(init_weights)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                AdaIN(channels=out_channels),
                ResBlock(
                    out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    leaky_relu_slope=leaky_relu_slope,
                ),
                AdaIN(channels=out_channels),
            )
            for kernel_size in kernel_sizes
        )

    def forward(self, x):
        x = self.input_conv(x)
        return torch.stack([block(x) for block in self.blocks], dim=0).mean(dim=0)

    def remove_weight_norm(self):
        for block in self.blocks:
            block[1].remove_weight_norm()


class SineGenerator(nn.Module):
    """The pitch as a sine wave at the output rate, plus noise."""

    def __init__(self, samp_rate, sine_amp=0.1, noise_std=0.003, voiced_threshold=0):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.merge = nn.Sequential(nn.Linear(1, 1, bias=False), nn.Tanh())

    def forward(self, f0, noise=None, phase0=None):
        """``f0`` is ``[batch, samples, 1]`` in Hz, 0 where unvoiced.

        ``noise`` (standard normal, one value per sample) and ``phase0`` (in
        cycles) replace the per-call noise and the zero starting phase.
        """
        with torch.no_grad():
            rad_values = (f0 / self.sampling_rate) % 1
            wrapped = torch.cumsum(rad_values, 1) % 1
            wrapped_over = (wrapped[:, 1:, :] - wrapped[:, :-1, :]) < 0
            cumsum_shift = torch.zeros_like(rad_values)
            cumsum_shift[:, 1:, :] = wrapped_over * -1.0
            phase = torch.cumsum(rad_values + cumsum_shift, dim=1)
            if phase0 is not None:
                phase = phase + phase0.to(phase.dtype).reshape(-1, 1, 1)
            sine_waves = torch.sin(phase * 2 * np.pi) * self.sine_amp
            uv = (f0 > self.voiced_threshold).to(sine_waves.dtype)
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            if noise is None:
                noise = torch.randn_like(sine_waves)
            else:
                noise = noise.to(sine_waves.dtype).reshape(sine_waves.shape)
            sine_waves = sine_waves * uv + noise_amp * noise
        return self.merge(sine_waves.to(self.merge[0].weight.dtype))


class RefineGANGenerator(nn.Module):
    """Turn the latent and the pitch into audio, guided by the sine at every scale."""

    def __init__(
        self,
        *,
        sample_rate,
        upsample_rates=(8, 8, 2, 2),
        leaky_relu_slope=0.2,
        num_mels=128,
        start_channels=16,
        gin_channels=256,
        upsample_initial_channel=512,
    ):
        super().__init__()
        self.upsample_rates = upsample_rates
        self.leaky_relu_slope = leaky_relu_slope
        self.upp = int(np.prod(upsample_rates))
        self.m_source = SineGenerator(sample_rate)
        self.pre_conv = weight_norm(nn.Conv1d(1, 16, 7, 1, padding=3))

        channels = start_channels
        size = self.upp
        self.downsample_blocks = nn.ModuleList()
        self.downsample_filters = nn.ModuleList()
        for i, _ in enumerate(upsample_rates):
            new_size = int(size / upsample_rates[-i - 1])
            self.downsample_filters.append(sinc_downsampler(size, new_size))
            size = new_size
            new_channels = channels * 2
            self.downsample_blocks.append(
                weight_norm(nn.Conv1d(channels, new_channels, 7, 1, padding=3))
            )
            channels = new_channels

        channels = upsample_initial_channel
        self.mel_conv = weight_norm(nn.Conv1d(num_mels, channels // 2, 7, 1, padding=3))
        self.mel_conv.apply(init_weights)
        if gin_channels != 0:
            self.cond = nn.Conv1d(256, channels // 2, 1)

        self.upsample_blocks = nn.ModuleList()
        self.upsample_conv_blocks = nn.ModuleList()
        for rate in upsample_rates:
            new_channels = channels // 2
            self.upsample_blocks.append(nn.Upsample(scale_factor=rate, mode="linear"))
            self.upsample_conv_blocks.append(
                ParallelResBlock(
                    in_channels=channels + channels // 4,
                    out_channels=new_channels,
                    kernel_sizes=(3, 7, 11),
                    dilation=(1, 3, 5),
                    leaky_relu_slope=leaky_relu_slope,
                )
            )
            channels = new_channels

        self.conv_post = weight_norm(nn.Conv1d(channels, 1, 7, 1, padding=3, bias=False))
        self.conv_post.apply(init_weights)

    def forward(self, x, f0, g=None, n_res=None, noise=None, phase0=None):
        if n_res is not None:
            n = int(n_res.item()) if isinstance(n_res, torch.Tensor) else int(n_res)
            if n != x.shape[-1]:
                x = F.interpolate(x, size=n, mode="linear")
        f0 = F.interpolate(f0.unsqueeze(1), size=x.shape[-1] * self.upp, mode="linear")
        source = self.m_source(f0.transpose(1, 2), noise, phase0).transpose(1, 2)
        y = self.pre_conv(source)
        downs = []
        for block, downsample in zip(self.downsample_blocks, self.downsample_filters):
            y = F.leaky_relu(y, self.leaky_relu_slope)
            downs.append(y)
            y = block(downsample(y.contiguous()))

        x = self.mel_conv(x)
        if g is not None:
            x = x + self.cond(g)
        x = torch.cat([x, y], dim=1)
        for ups, res, down in zip(
            self.upsample_blocks, self.upsample_conv_blocks, reversed(downs)
        ):
            x = F.leaky_relu(x, self.leaky_relu_slope)
            x = res(torch.cat([ups(x), down], dim=1))
        x = F.leaky_relu(x, self.leaky_relu_slope)
        return torch.tanh(self.conv_post(x))

    def remove_weight_norm(self):
        remove_weight_norm(self.pre_conv)
        remove_weight_norm(self.mel_conv)
        remove_weight_norm(self.conv_post)
        for block in self.downsample_blocks:
            remove_weight_norm(block)
        for block in self.upsample_conv_blocks:
            block.remove_weight_norm()
