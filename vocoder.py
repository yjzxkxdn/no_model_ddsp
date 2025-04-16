import torch
import torch.nn as nn
import torch.nn.functional as F
from ddsp.core import frequency_filter, mean_filter, upsample

class Sins(nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            win_length,
            use_mean_filter,
            n_harmonics,
            n_mag_noise,
            prediction_phase,
            batch,
            n_frames,
            device,
            amplitudes = None,
            harmonic_phase = None,
            noise_magnitude = None,
            noise_phase = None,
            f0_frames = None
        ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.block_size = block_size
        self.win_length = win_length
        self.window = torch.hann_window(win_length, device=device)
        self.prediction_phase = prediction_phase
        self.device = device
        
        # Parameters with correct device handling
        if amplitudes is None:
            self.amplitudes = nn.Parameter(
                torch.zeros(batch, n_frames, n_harmonics, device=device) * 0.1,
                requires_grad=True
            )
        else:
            self.amplitudes = nn.Parameter(amplitudes, requires_grad=True)
        
        if prediction_phase:
            if harmonic_phase is None:
                self.harmonic_phase = nn.Parameter(
                    torch.zeros(batch, n_frames, win_length // 2 + 1, device=device) * 0.1,
                    requires_grad=True
                )
            else:
                self.harmonic_phase = nn.Parameter(harmonic_phase, requires_grad=True)
        else:
            self.harmonic_phase = None
            
        if noise_magnitude is None:
            self.noise_magnitude = nn.Parameter(
                torch.zeros(batch, n_frames, n_mag_noise, device=device) * 0.1,
                requires_grad=True
            )
        else:
            self.noise_magnitude = nn.Parameter(noise_magnitude, requires_grad=True)
            
        if noise_phase is None:
            self.noise_phase = nn.Parameter(
                torch.zeros(batch, n_frames, n_mag_noise, device=device) * 0.1,
                requires_grad=True
            )
        else:
            self.noise_phase = nn.Parameter(noise_phase, requires_grad=True)
            
        if f0_frames is not None:
            self.f0 = nn.Parameter(torch.log(f0_frames), requires_grad=True)
        else:
            self.f0 = None
            
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

    def forward(self, f0_frames = None, max_upsample_dim=32):
        if f0_frames is None:
            f0_frames = self.f0
        # exciter phase
        phase = self.fast_phase_gen(f0_frames)
        sinusoid = torch.sin(phase).squeeze(-1)
        
        # noise exciter signal
        noise = torch.randn_like(sinusoid)
        
        if self.mean_kernel_size > 1:
            amplitudes = mean_filter(self.amplitudes, self.mean_kernel_size)
            if self.prediction_phase:
                harmonic_phase = mean_filter(self.harmonic_phase, self.mean_kernel_size)
        else:
            amplitudes = self.amplitudes
            harmonic_phase = self.harmonic_phase

        amplitudes_frames = torch.exp(amplitudes)/ 128
        noise_param = torch.exp(self.noise_magnitude + 1.j * torch.pi * self.noise_phase) / 128
        
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
        
        if self.prediction_phase:
            src_allpass = torch.exp(1.j * torch.pi * harmonic_phase)
            src_allpass = torch.cat((src_allpass, src_allpass[:,-1:,:]), 1)
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

        return signal, sinusoids, (harmonic, noise), f0_frames
    
class CombSub(torch.nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            win_length,
            use_mean_filter,
            n_mag_harmonic,
            n_mag_noise,
            prediction_phase,
            batch,
            n_frames,
            device,
            harmonic_magnitude = None,
            harmonic_phase = None,
            noise_magnitude = None,
            noise_phase = None,
            f0_frames = None
        ):
        super().__init__()

        print(' [DDSP Model] Combtooth Subtractive Synthesiser')
        # params
        self.sampling_rate = sampling_rate
        self.block_size = block_size
        self.win_length = win_length
        self.window = torch.hann_window(win_length, device=device)
        self.device = device
        self.prediction_phase = prediction_phase
        
        # Parameters with correct device handling
        if harmonic_magnitude is None:
            self.harmonic_magnitude = nn.Parameter(
                torch.zeros(batch, n_frames, n_mag_harmonic, device=device) * 0.1,
                requires_grad=True
            )
        else:
            self.harmonic_magnitude = nn.Parameter(harmonic_magnitude, requires_grad=True)
        if prediction_phase:
            if harmonic_phase is None:
                self.harmonic_phase = nn.Parameter(
                    torch.zeros(batch, n_frames, win_length // 2 + 1, device=device) * 0.1,
                    requires_grad=True
                )
            else:
                self.harmonic_phase = nn.Parameter(harmonic_phase, requires_grad=True)
        else:
            self.harmonic_phase = None
        
        if noise_magnitude is None:
            self.noise_magnitude = nn.Parameter(
                torch.zeros(batch, n_frames, n_mag_noise, device=device) * 0.1,
                requires_grad=True
            )
        else:
            self.noise_magnitude = nn.Parameter(noise_magnitude, requires_grad=True)
        if noise_phase is None:
            self.noise_phase = nn.Parameter(
                torch.zeros(batch, n_frames, n_mag_noise, device=device) * 0.1,
                requires_grad=True
            )
        if f0_frames is not None:
            #self.f0 = nn.Parameter(f0_frames, requires_grad=True)
            self.log_f0 = nn.Parameter(torch.log(f0_frames), requires_grad=True)
        else:
            self.log_f0 = None
            
        

        # mean filter kernel size
        if use_mean_filter:
            self.mean_kernel_size = win_length // block_size
        else:
            self.mean_kernel_size = 1
    
    def fast_source_gen(self, f0_frames):
        n = torch.arange(self.block_size, device=f0_frames.device)
        s0 = f0_frames / self.sampling_rate
        ds0 = F.pad(s0[:, 1:, :] - s0[:, :-1, :], (0, 0, 0, 1))
        rad = s0 * (n + 1) + 0.5 * ds0 * n * (n + 1) / self.block_size
        s0 = s0 + ds0 * n / self.block_size
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0_frames)
        rad += F.pad(rad_acc[:, :-1, :], (0, 0, 1, 0))
        rad -= torch.round(rad)
        combtooth = torch.sinc(rad / (s0 + 1e-5)).reshape(f0_frames.shape[0], -1)
        return combtooth
        
    def forward(self, 
            f0_frames=None,
            **kwargs):
        '''
            mel_frames: B x n_frames x n_mels
            f0_frames: B x n_frames x 1
        '''
        if f0_frames is None:
            f0_frames = torch.exp(self.log_f0)
        # combtooth exciter signal
        combtooth = self.fast_source_gen(f0_frames)
        
        # noise exciter signal
        noise = torch.randn_like(combtooth)
        
        # parameter prediction
        if self.mean_kernel_size > 1:
            harmonic_magnitude = mean_filter(self.harmonic_magnitude, self.mean_kernel_size)
            if self.prediction_phase:
                harmonic_phase = mean_filter(self.harmonic_phase, self.mean_kernel_size)
        else:
            harmonic_magnitude = self.harmonic_magnitude
            harmonic_phase = self.harmonic_phase
        
        src_param = torch.exp(harmonic_magnitude)
        noise_param = torch.exp(self.noise_magnitude + 1.j * torch.pi * self.noise_phase) / 128
        
        harmonic = frequency_filter(
                        combtooth,
                        torch.complex(src_param, torch.zeros_like(src_param)),
                        hann_window = True,
                        half_width_frames = 1.5 * self.sampling_rate / (f0_frames + 1e-3))
               
        if self.prediction_phase:   
            src_allpass = torch.exp(1.j * torch.pi * harmonic_phase)
            src_allpass = torch.cat((src_allpass, src_allpass[:,-1:,:]), 1)
            # harmonic part filter (all pass)
            harmonic_spec = torch.stft(
                                harmonic,
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

        return signal, combtooth, (harmonic, noise), f0_frames
    
