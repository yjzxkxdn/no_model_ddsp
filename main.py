import torch
import numpy as np
import parselmouth as pm
import soundfile as sf
import matplotlib.pyplot as plt
import yaml

from vocoder import Sins, CombSub
from loss import HybridLoss
from utils import interp_f0, expand_uv

class DotDict(dict):
    def __getattr__(self, attr):
        return self.get(attr)
    def __setattr__(self, key, value):
        self[key] = value
        
def extract_f0_parselmouth(config, x: np.ndarray, n_frames):
    l_pad = int(
            np.ceil(
                1.5 / config.f0_min * config.sampling_rate
            )
    )
    r_pad = config.block_size * ((len(x) - 1) // config.block_size + 1) - len(x) + l_pad + 1
    padded_signal = np.pad(x, (l_pad, r_pad))
    
    sound = pm.Sound(padded_signal, config.sampling_rate)
    pitch = sound.to_pitch_ac(
        time_step=config.block_size / config.sampling_rate, 
        voicing_threshold=0.6,
        pitch_floor=config.f0_min, 
        pitch_ceiling=1100
    )
    
    f0 = pitch.selected_array['frequency']
    if len(f0) < n_frames:
        f0 = np.pad(f0, (0, n_frames - len(f0)))
    f0 = f0[:n_frames]

    return f0

def main(input_audio, config, device):
    """
    Args:
        input_audio (Tensor): B x T
        config (DotDict): config object
        device (str): cuda or cpu
    Returns:
        Tensor: B x T
    """
    n_frames = input_audio.shape[1] // config.block_size
    print(f"Number of frames: {n_frames}")
    input_audio = input_audio[:, :n_frames * config.block_size] # 切分为 block_size 的整数倍长度

    f0 = extract_f0_parselmouth(config, input_audio.squeeze().numpy(), n_frames=n_frames)
    f0, uv = interp_f0(f0)
    uv = expand_uv(uv)
    
    f0 = torch.from_numpy(f0).unsqueeze(0).unsqueeze(-1)
    uv = torch.from_numpy(uv).unsqueeze(0)
    
    print("F0 shape:", f0.shape)
    print("UV shape:", uv.shape)
    
    input_audio = input_audio.to(device)
    f0 = f0.to(device)
    uv = uv.to(device)

    batch = input_audio.shape[0]

    if config.model == "Sins":
        model = Sins(
            sampling_rate=config.sampling_rate,
            block_size=config.block_size,
            win_length=config.win_length,
            use_mean_filter=config.use_mean_filter,
            n_harmonics=config.n_harmonics,
            n_mag_noise=config.n_mag_noise,
            prediction_phase = config.prediction_phase,
            batch=batch,
            n_frames=n_frames,
            device=device
        )
    elif config.model == "CombSub":
        model = CombSub(
            sampling_rate=config.sampling_rate,
            block_size=config.block_size,
            win_length=config.win_length,
            use_mean_filter=config.use_mean_filter,
            n_mag_harmonic=config.n_mag_harmonic,
            n_mag_noise=config.n_mag_noise,
            prediction_phase = config.prediction_phase,
            batch=batch,
            n_frames=n_frames,
            device=device
        )

    lr = config.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    
    criterion = HybridLoss(
        config.block_size,
        config.fft_min,
        config.fft_max,
        config.n_scale,
        config.lambda_uv,
        device
    )

    # 训练循环
    num_epochs = config.num_epochs
    for epoch in range(num_epochs):

        optimizer.zero_grad()

        output_signal, sinusoids, (harmonic, noise), f0_pred = model(f0_frames = f0)  

        loss, loss_rss, loss_uv = criterion(output_signal, harmonic, input_audio, uv)

        loss.backward()

        optimizer.step()
        scheduler.step()


        print(f"Epoch [{epoch+1}/{num_epochs}], Lr: {optimizer.param_groups[0]['lr']:.4f}, Loss: {loss.item():.4f}, RSS Loss: {loss_rss.item():.4f}, UV Loss: {loss_uv.item():.4f}")

    return output_signal, sinusoids, (harmonic, noise), f0, f0_pred

if __name__ == '__main__':

    input_audio, sr = sf.read(r"约定vocal.wav")
    config_path = r"CombSub.yaml"
    input_audio = torch.from_numpy(input_audio).unsqueeze(0)
    config = DotDict(yaml.load(open(config_path, "r"), Loader=yaml.FullLoader))
    print(config)

    output_audio, _, (harmonic, noise), f0, f0_pred = main(input_audio, config, device=config.device)

    sf.write("output.wav", output_audio.detach().squeeze().cpu().numpy(), sr)
    sf.write("harmonic.wav", harmonic.detach().squeeze().cpu().numpy(), sr) 
    sf.write("noise.wav", noise.detach().squeeze().cpu().numpy(), sr) 
    
    # 绘制 F0 图
    plt.figure(figsize=(10, 5))
    plt.plot(f0.detach().squeeze().cpu().numpy())

    
    plt.legend([ 'Ground Truth'])
    plt.title('F0 Comparison')
    plt.xlabel('Frame')
    plt.ylabel('Frequency (Hz)')

    plt.figure(figsize=(10, 5))
    plt.plot(f0_pred.detach().squeeze().cpu().numpy())
    plt.legend([ 'Predicted'])
    plt.title('F0 Comparison')
    plt.xlabel('Frame')
    plt.ylabel('Frequency (Hz)')
    plt.show()
    

    
    