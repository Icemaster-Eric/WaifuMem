from TTS.api import TTS


class RaineTTS:
    def __init__(self):
        self.tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    
    def tts(self, text: str, ref_wav: str, language: str = "en"):
        self.tts_model.tts_to_file(text, speaker_wav=ref_wav, language=language, split_sentences=True)


if __name__ == "__main__":
    tts = RaineTTS()

    tts.tts(
        "Ah... back when Beidou and I were carrying out ambushes against the Shogun's Army together, I raised my concerns that the enemy might detect our approach. But Beidou assured me of her fleet's unwavering discipline, and that raids could be carried out without detection. In the end, she was right, and that's exactly what happened. Now, every time the Watatsumi fleet encounters some sort of bottleneck, I consider inviting her to come and give them her instruction. There's no doubt, she runs a tight ship...",
        ref_wav="tts/ref.wav"
    )
