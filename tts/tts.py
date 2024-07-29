import re
import os
import sounddevice as sd
from styletts2 import tts
from tts.rvc.infer.modules.vc.modules import VC
from tts.rvc.configs.config import Config


# I give up
os.environ["rmvpe_root"] = "tts/models"


class RaineTTS:
    def __init__(self):
        self.tts_model = tts.StyleTTS2()
        self.rvc_config = Config()
        self.rvc_config.n_cpu = 2
        self.rvc_model = VC(self.rvc_config)
        self.rvc_model.get_vc("tts/models/kokomi-rvc2.pth", "tts/models/kokomi/kokomi-rvc2.index")
        self.rvc_model.load_hubert("tts/models/hubert_base.pt")

    def tts(self, text: str):
        pattern = "|".join(f"(?<={re.escape(delim)})" for delim in (". ", "? ", "! "))

        for sentence in re.split(pattern, text):
            self.tts_model.inference(sentence, output_wav_file="output.wav")

            output = self.rvc_model.vc_single(
                0,
                "output.wav",
                3,
                0,
                "rmvpe",
                "tts/models/kokomi/kokomi-rvc2.index",
                None,
                0,
                0.88,
                0.33,
                0.5,
                0.33
            )

            sd.play(output[1][1], output[1][0])
            sd.wait()
