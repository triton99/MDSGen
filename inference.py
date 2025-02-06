import os
import torch
import numpy as np
import random
import argparse
import soundfile as sf

from diffusers.utils.torch_utils import randn_tensor
from diffusers import AudioLDMPipeline

from masked_diffusion.models import MDSGen_B_2
from masked_diffusion.gaussian_diffusion import get_named_beta_schedule
from masked_diffusion.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver


def set_global_seed(seed):
    np.random.seed(seed % (2**32))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="./ckpts/mdsgen_audioldm.pt", help="checkpoint path")
    parser.add_argument("--save_path", type=str, default="./output_wav", help="save folder path")
    parser.add_argument("--video_feat_path", type=str, default="./sources/__2MwJ2uHu0_000004.npz", help="video feature path")
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    return parser

###############################################################################################

def main():
    parser = get_args_parser()
    args = parser.parse_args()

    # Set seed
    os.makedirs(args.save_path, exist_ok=True)
    set_global_seed(args.seed)

    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_scale = 5.0
    pow_scale = 0.01
    num_sampling_steps = 50

    sr = 16000
    truncate = 130560
    fps = 4
    truncate_frame = int(fps * truncate / sr)

    # Load model
    model = MDSGen_B_2(decode_layer=4).to(device)
    state_dict = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model.eval()

    # Load VAE and Vocoder
    pipe = AudioLDMPipeline.from_pretrained("cvssp/audioldm-s-full-v2")
    scale_factor = pipe.vae_scale_factor
    vae = pipe.vae.to(device)
    vae.eval()
    vocoder = pipe.vocoder.to(device)

    def model_forward(x, t, cfg_scale=None, diffusion_steps=1000, scale_pow=4.0, mix_video_feat=None):
        C = 8
        model_output = model.forward_with_cfg(x, t, cfg_scale, diffusion_steps, scale_pow, mix_video_feat)
        return torch.split(model_output, C, dim=1)[0]
    betas = get_named_beta_schedule("linear", 1000)
    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.from_numpy(betas))

    # Load video feature
    output_names = os.path.basename(args.video_feat_path).split(".")[0]
    video_feat = np.load(args.video_feat_path)['arr_0'].astype(np.float32)
    video_feat = torch.from_numpy(video_feat[:truncate_frame]).contiguous().to(device)
    video_feat = video_feat.unsqueeze(0)

    weight_dtype = torch.bfloat16 # "fp16"
    generator = torch.manual_seed(args.seed)

    latent_size = (204, 16)
    shape = (len(video_feat), 8, latent_size[0], latent_size[1])
    z = randn_tensor(shape, generator=generator, device=video_feat.device, dtype=weight_dtype)

    model_kwargs = dict(cfg_scale=cfg_scale, scale_pow=pow_scale,
                        mix_video_feat=video_feat,
                        )

    algorithm_type = 'dpmsolver++'
    method = 'multistep'
    order = 2
    skip_type = 'time_uniform'

    model_fn = model_wrapper(
        model_forward,
        noise_schedule,
        model_type="noise",
        model_kwargs=model_kwargs,
    )

    dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type=algorithm_type)

    # Run Inference
    samples = dpm_solver.sample(
        z,
        steps=num_sampling_steps,
        order=order,
        skip_type=skip_type,
        method=method,
    )

    # Decode and vocode
    output_mels = vae.decode(samples / scale_factor).sample
    wav_samples = vocoder(output_mels.squeeze())
    wav_sample = wav_samples.detach().cpu().numpy()

    sf.write(os.path.join(args.save_path, output_names + ".wav"), wav_sample, sr)

    print("========================================FINISH INFERENCE===========================================")

if __name__ == "__main__":
    main()

