from TTS.api import TTS


class RaineTTS:
    def __init__(self):
        self.tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

    def tts(self, text: str, ref_wav: str, language: str = "en"):
        self.tts_model.tts_to_file(text, speaker_wav=ref_wav, language=language, split_sentences=True)


if __name__ == "__main__":
    tts = RaineTTS()

    tts.tts(
        """I don't think SeamlessM4T qualifies as an end-to-end audio-to-audio model. The paper states "the task of speech-to-speech translation in SeamlessM4T v2 is broken down into speech-to-text translation (S2TT) and then text-to-unit conversion (T2U)". And while language translation is an important application as you mention, it's strictly limited to that. It wouldn't understand or produce non-speech audio (e.g. singing, music, environmental sounds, etc) and you can't have a conversation with it.""",
        ref_wav="tts/ref.wav"
    )
