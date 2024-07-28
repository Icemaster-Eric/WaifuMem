from styletts2 import tts
from scipy.io.wavfile import write
from tts.rvc.misc import load_hubert, get_vc, vc_single


class RaineTTS:
    def __init__(self):
        self.hubert_model = load_hubert("tts/models/hubert_base.pt")
        self.tts_model = tts.StyleTTS2()

    def tts(self, text: str):
        self.tts_model.inference(text, output_wav_file="output.wav")

        get_vc("kokomi-rvc2.pth", "tts/models/kokomi", 0.33, 0.5)
        wav_opt = vc_single(
            0, 
            "output.wav",
            3, 
            None, 
            "harvest", 
            "kokomi-rvc2",
            '',
            0.88,
            3,
            0,
            1,
            0.33,
        )

        write("rvc-output.mp3", wav_opt[1][0], wav_opt[1][1])
