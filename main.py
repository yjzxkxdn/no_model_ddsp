
import torch
import soundfile as sf
import matplotlib.pyplot as plt
import yaml

from vocoder import Sins, CombSub
from loss import HybridLoss
from utils import interp_f0, expand_uv, extract_f0_parselmouth, DotDict, analyze_model_parameters


def main(input_audio, f0, uv, config, device,
        harmonic_magnitude = None,
        harmonic_phase = None,
        noise_magnitude = None,
        noise_phase = None
    ):
    """
    Args:
        input_audio (Tensor): B x T
        config (DotDict): config object
        device (str): cuda or cpu
    Returns:
        Tensor: B x T
    """

    batch = input_audio.shape[0]

    #f0[:, 500:700, : ] *= 2 # 模拟一处倍频错误
    
    if config.prediction_f0:
        f0_frames_input = f0.clone()
    else:
        f0_frames_input = None
        
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
            device=device,
            amplitudes = harmonic_magnitude,
            harmonic_phase = harmonic_phase,
            noise_magnitude = noise_magnitude,
            noise_phase = noise_phase,
            f0_frames = f0_frames_input,
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
            device=device,
            harmonic_magnitude = harmonic_magnitude,
            harmonic_phase = harmonic_phase,
            noise_magnitude = noise_magnitude,
            noise_phase = noise_phase,
            f0_frames = f0_frames_input,
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
    
    print("Optimizer parameters:", [name for name, param in model.named_parameters()])

    # 训练循环
    num_epochs = config.num_epochs
    for epoch in range(num_epochs):

        optimizer.zero_grad()

        if config.prediction_f0:
            output_signal, sinusoids, (harmonic, noise), f0_pred = model()  
        else:
            output_signal, sinusoids, (harmonic, noise), f0_pred = model(f0)  

        loss, loss_rss, loss_uv = criterion(output_signal, harmonic, input_audio, uv)

        loss.backward()
        
        optimizer.step()
        scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Lr: {optimizer.param_groups[0]['lr']:.4f}, Loss: {loss.item():.4f}, RSS Loss: {loss_rss.item():.4f}, UV Loss: {loss_uv.item():.4f}")
    analyze_model_parameters(model)
    
    return model

if __name__ == '__main__':

    input_audio, sr = sf.read(r"约定vocal.wav")
    config_path = r"CombSub.yaml"
    
    
    
    config = DotDict(yaml.load(open(config_path, "r"), Loader=yaml.FullLoader))
    print(config)
    device = config.device
    
    n_frames = input_audio.shape[0] // config.block_size
    input_audio = input_audio[:n_frames * config.block_size] # 切分为 block_size 的整数倍长度
    input_audio_t = torch.from_numpy(input_audio).unsqueeze(0)
    
    f0 = extract_f0_parselmouth(config, input_audio, n_frames=n_frames)
    f0, uv = interp_f0(f0)
    uv = expand_uv(uv)
    
    f0 = torch.from_numpy(f0).unsqueeze(0).unsqueeze(-1)
    uv = torch.from_numpy(uv).unsqueeze(0)
    
    print("F0 shape:", f0.shape)
    print("UV shape:", uv.shape)
    
    input_audio_t = input_audio_t.to(device)
    f0 = f0.to(device)
    uv = uv.to(device)
    
    model = main(input_audio_t, f0, uv, config, device=config.device)
    
    output_audio, _, (harmonic, noise), f0_pred = main(input_audio, config, device=config.device)
    sf.write("output.wav", output_audio.detach().squeeze().cpu().numpy(), sr)
    sf.write("harmonic.wav", harmonic.detach().squeeze().cpu().numpy(), sr) 
    sf.write("noise.wav", noise.detach().squeeze().cpu().numpy(), sr) 
    
    # 绘制 F0 图
    plt.figure(figsize=(10, 5))
    plt.plot(f0.detach().squeeze().cpu().numpy())
    plt.legend([ 'Original'])
    plt.title('F0 Comparison')
    plt.xlabel('Frame')
    plt.ylabel('Frequency (Hz)')

    plt.figure(figsize=(10, 5))
    plt.plot(f0_pred.detach().squeeze().cpu().numpy())
    plt.legend([ 'Predicted'])
    plt.title('F0 Comparison')
    plt.xlabel('Frame')
    plt.ylabel('Frequency (Hz)')
    
    plt.figure(figsize=(10, 5))
    plt.plot(f0.detach().squeeze().cpu().numpy())
    plt.plot(f0_pred.detach().squeeze().cpu().numpy())
    plt.legend([ 'Original', 'Predicted'])
    plt.title('F0 Comparison')
    plt.xlabel('Frame')
    plt.ylabel('Frequency (Hz)')
    plt.show()
    

    
    