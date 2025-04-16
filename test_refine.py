import torch
import soundfile as sf
import yaml
from ddsp.vocoder import load_model, Audio2Mel
from utils import interp_f0, expand_uv, extract_f0_parselmouth, DotDict, analyze_model_parameters
from main import main

if __name__ == '__main__':
    
    input_audio, sr = sf.read(r"约定vocal.wav")
    config_path = r"CombSub.yaml"
    ddsp_model_path = r""
    
    config = DotDict(yaml.load(open(config_path, "r"), Loader=yaml.FullLoader))
    print(config)
    device = config.device
    
    n_frames = input_audio.shape[0] // config.block_size
    input_audio = input_audio[:n_frames * config.block_size] # 切分为 block_size 的整数倍长度
    input_audio_t = torch.from_numpy(input_audio).unsqueeze(0) #[1, n_frames * block_size]
    
    # extract f0
    f0 = extract_f0_parselmouth(config, input_audio, n_frames=n_frames)
    f0, uv = interp_f0(f0)
    uv = expand_uv(uv)
    
    f0 = torch.from_numpy(f0).unsqueeze(0).unsqueeze(-1)
    uv = torch.from_numpy(uv).unsqueeze(0)
    
    # load model
    model, args = load_model(ddsp_model_path, device=device)
    
    sampling_rate = args.data.sampling_rate
    hop_length = args.data.block_size
    win_length = args.data.win_length
    n_fft = args.data.n_fft
    n_mel_channels = args.data.n_mels
    mel_fmin = args.data.mel_fmin
    mel_fmax = args.data.mel_fmax
    
    assert args.model.type == config.model
    
    # mel analysis
    mel_extractor = Audio2Mel(
        hop_length=hop_length,
        sampling_rate=sampling_rate,
        n_mel_channels=n_mel_channels,
        win_length=win_length,
        n_fft=n_fft,
        mel_fmin=mel_fmin,
        mel_fmax=mel_fmax
    ).to(device)
    
    mel = mel_extractor(input_audio_t.unsqueeze(0))
    
     
    # forward and save the output
    with torch.no_grad():
        harmonic_magnitude, harmonic_phase, noise_magnitude, noise_phase = model(mel, f0)

    model_refine = main(
        input_audio_t, f0, uv, config, device=config.device,
        harmonic_magnitude = harmonic_magnitude.clone(),
        harmonic_phase = harmonic_phase.clone(),
        noise_magnitude = noise_magnitude.clone(),
        noise_phase = noise_phase.clone(),
    )
    
    output_audio, _, (harmonic, noise), _ = model_refine(f0)
    sf.write("output_refine.wav", output_audio.detach().squeeze().cpu().numpy(), sr)
    sf.write("harmonic_refine.wav", harmonic.detach().squeeze().cpu().numpy(), sr) 
    sf.write("noise_refine.wav", noise.detach().squeeze().cpu().numpy(), sr) 

