import edge_tts
from scipy.io.wavfile import write
from tts.rvc.misc import load_hubert, get_vc, vc_single


class RaineTTS:
    def __init__(self):
        self.hubert_model = load_hubert("tts/models/hubert_base.pt")

    def tts(self, text: str):
        #get_vc(speaker_id, rvc_model_dir, 0.33, 0.5)
        pass


if __name__ == "__main__":
    raine_tts = RaineTTS()

    raine_tts.tts("Woah, so this is my new voice? Pretty cool.")
