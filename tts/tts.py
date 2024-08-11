import os
import sounddevice as sd
from scipy.io.wavfile import write
import soundfile as sf
import numpy as np
from pysbd import Segmenter
from tts.rvc.infer.modules.vc.modules import VC
from tts.rvc.configs.config import Config


os.environ["rmvpe_root"] = "tts/models"


class RaineTTS:
    def __init__(self):
        self.rvc_config = Config()
        self.rvc_config.n_cpu = 2
        self.rvc_model = VC(self.rvc_config)
        self.rvc_model.get_vc("tts/models/kokomi-rvc2.pth", "tts/models/kokomi/kokomi-rvc2.index")
        self.rvc_model.load_hubert("tts/models/hubert_base.pt")

        self.segmenter = Segmenter()

    def tts(self, text: str):
        audio = []

        for sentence in self.segmenter.segment(text):
            pass

        output = self.rvc_model.vc_single(
            0,
            "tts-output.wav",
            2,
            0,
            "rmvpe",
            "tts/models/kokomi/kokomi-rvc2.index",
            None,
            0,
            0.88,
            0.33,
            0.5,
            0.66
        )

        write("rvc-output.wav", output[1][0], output[1][1])

        #sd.play(output[1][1], output[1][0])
        #sd.wait()
