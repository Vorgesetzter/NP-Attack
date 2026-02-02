import os
import hydra
import soundfile as sf
import torch
from omegaconf import DictConfig, OmegaConf

from models.attacker import NPAttacker

# Override the input wave file here (set to None to use config default)
_WAVE_FILE = "ground_truth.wav"  # e.g., "data/my_audio.wav" or "/absolute/path/to/file.flac"
# Convert to absolute path before Hydra changes working directory
WAVE_FILE = os.path.abspath(_WAVE_FILE) if _WAVE_FILE is not None else None


@hydra.main(config_path="conf", config_name="config")
def main(args: DictConfig):
    print(OmegaConf.to_yaml(args))
    torch.manual_seed(args.seed)

    wave_file = WAVE_FILE if WAVE_FILE is not None else args.wave_file

    model = NPAttacker(args)
    wave = model.attack(wave_file)

    if wave is not None:
        # retest the output
        out = args.strategy.name + str(args.seed) + '.wav'
        sf.write(out, wave, args.sr)
        print(model.asr.model.transcribe_file(out))

        if args.out:
            model.eval_attack(wave)

if __name__ == '__main__':
    main()
