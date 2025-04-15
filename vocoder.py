import torch
import torch.nn as nn
import torch.nn.functional as F
from core import frequency_filter, mean_filter, upsample

class Sins(nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            win_length,
            use_mean_filter,
            n_harmonics,
            n_mag_noise,
            batch,
            n_frames,
            device
        ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.block_size = block_size
        self.win_length = win_length
        self.window = torch.hann_window(win_length, device=device)
        self.device = device
        
        # Parameters with correct device handling
        self.amplitudes = nn.Parameter(
            torch.randn(batch, n_frames, n_harmonics, device=device) * 0.1,
            requires_grad=True
        )
        self.harmonic_phase = nn.Parameter(
            torch.randn(batch, n_frames, win_length // 2 + 1, device=device) * 0.1,
            requires_grad=True
        )
        self.noise_amplitude = nn.Parameter(
            torch.randn(batch, n_frames, n_mag_noise, device=device) * 0.1,
            requires_grad=True
        )
        self.noise_phase = nn.Parameter(
            torch.randn(batch, n_frames, n_mag_noise, device=device) * 0.1,
            requires_grad=True
        )
        
        # mean filter kernel size
        if use_mean_filter:
            self.mean_kernel_size = win_length // block_size
        else:
            self.mean_kernel_size = 1
            
    def fast_phase_gen(self, f0_frames):
        n = torch.arange(self.block_size, device=f0_frames.device)
        s0 = f0_frames / self.sampling_rate
        ds0 = F.pad(s0[:, 1:, :] - s0[:, :-1, :], (0, 0, 0, 1))
        rad = s0 * (n + 1) + 0.5 * ds0 * n * (n + 1) / self.block_size
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0_frames)
        rad += F.pad(rad_acc[:, :-1, :], (0, 0, 1, 0))
        phase = 2 * torch.pi * rad.reshape(f0_frames.shape[0], -1, 1)
        return phase

    def forward(self, f0_frames, max_upsample_dim=32):
        # exciter phase
        phase = self.fast_phase_gen(f0_frames)
        sinusoid = torch.sin(phase).squeeze(-1)
        
        # noise exciter signal
        noise = torch.randn_like(sinusoid)
        
        if self.mean_kernel_size > 1:
            self.amplitudes = mean_filter(self.amplitudes, self.mean_kernel_size)
            self.harmonic_phase = mean_filter(self.harmonic_phase, self.mean_kernel_size)

        src_allpass = torch.exp(1.j * torch.pi * self.harmonic_phase)
        src_allpass = torch.cat((src_allpass, src_allpass[:,-1:,:]), 1)
        amplitudes_frames = torch.exp(self.amplitudes)/ 128
        noise_param = torch.exp(self.noise_amplitude + 1.j * torch.pi * self.noise_phase) / 128
        
        n_harmonic = amplitudes_frames.shape[-1]
        level_harmonic = torch.arange(1, n_harmonic + 1, device=phase.device)
        mask = (f0_frames * level_harmonic < self.sampling_rate / 2).float() + 1e-7
        amplitudes_frames *= mask
        sinusoids = 0.
        for n in range(( n_harmonic - 1) // max_upsample_dim + 1):
            start = n * max_upsample_dim
            end = (n + 1) * max_upsample_dim
            phases = phase * level_harmonic[start:end]
            amplitudes = upsample(amplitudes_frames[:,:,start:end], self.block_size)
            sinusoids += (torch.sin(phases) * amplitudes).sum(-1)
        
        # harmonic part filter (all pass)
        harmonic_spec = torch.stft(
                            sinusoids,
                            n_fft = self.win_length,
                            win_length = self.win_length,
                            hop_length = self.block_size,
                            window = self.window,
                            center = True,
                            return_complex = True)
        harmonic_spec = harmonic_spec * src_allpass.permute(0, 2, 1)
        harmonic = torch.istft(
                        harmonic_spec,
                        n_fft = self.win_length,
                        win_length = self.win_length,
                        hop_length = self.block_size,
                        window = self.window,
                        center = True)
                        
        # noise part filter (using constant-windowed LTV-FIR) 
        noise = frequency_filter(
                        noise,
                        noise_param)
                        
        signal = harmonic + noise

        return signal, sinusoids, (harmonic, noise)