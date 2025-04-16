import torch
import soundfile as sf
import yaml

from utils import interp_f0, expand_uv, extract_f0_parselmouth, DotDict, analyze_model_parameters
from main import main

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
    
    for i in [1,1.5,2.0,2.5,0.8,0.3,0.1]:
        output_audio, _, (harmonic, noise), f0_pred = model(f0*i)
        sf.write(f"output_{+i}.wav", output_audio.detach().squeeze().cpu().numpy(), sr)
        sf.write(f"harmonic_{+i}.wav", harmonic.detach().squeeze().cpu().numpy(), sr) 
        sf.write(f"noise_{+i}.wav", noise.detach().squeeze().cpu().numpy(), sr) 
    
    

    
    