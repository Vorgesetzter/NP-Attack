import torch
import whisper
import string


class ASR:
    def __init__(self, hp, device) -> None:
        self.hp = hp
        self.device = device
        # You can use hp.source from your config to specify size (e.g., "base", "small")
        # Or hardcode it here.
        model_name = "base"

        # If your config's hp.source is a valid whisper size, use it:
        # valid_models = ["tiny", "base", "small", "medium", "large"]
        # if hp.source in valid_models:
        #     model_name = hp.source

        self.model = whisper.load_model(model_name, device=device)

    def transcribe(self, wave):
        """
        Args:
            wave (np.array): Audio data (16kHz, float32)
        """
        # Convert to tensor and pad/trim to 30 seconds
        audio_tensor = torch.from_numpy(wave).float()
        audio_tensor = whisper.pad_or_trim(audio_tensor)

        # Create Mel spectrogram
        mel = whisper.log_mel_spectrogram(audio_tensor, n_mels=self.model.dims.n_mels).to(self.device)

        # Use decode() with temperature=0 for deterministic greedy decoding
        # without_timestamps reduces hallucination on padded silence
        decode_options = whisper.DecodingOptions(
            without_timestamps=True,
            temperature=0,
            language="en"
        )
        result = whisper.decode(self.model, mel, decode_options)

        text = result.text

        # Normalize: uppercase and remove punctuation
        text = text.upper()
        text = text.translate(str.maketrans('', '', string.punctuation))

        return text.strip()