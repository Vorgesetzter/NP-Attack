import torch
import whisper
import re


class ASR:
    def __init__(self, hp, device) -> None:
        self.hp = hp
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
        # Whisper's .transcribe() handles moving to GPU and padding internally.
        # We force English to ensure consistent attacking.
        result = self.model.transcribe(wave, language="en", fp16=False)
        text = result["text"]

        # IMPORTANT: Normalize Whisper output to match standard ASR benchmarks.
        # Whisper outputs: "Hello, world."
        # Standard ASR expects: "HELLO WORLD"
        # If we don't do this, the WER calculation will be huge due to punctuation.
        text = text.upper()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation (.,?,!)

        return text.strip()