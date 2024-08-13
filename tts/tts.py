import os
import sounddevice as sd
import torch
from scipy.io.wavfile import write
import soundfile as sf
import numpy as np
from edge_tts import Communicate
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

    def tts(self, text: str):
        #communicate = Communicate(text, "en-US-AriaNeural")
        #communicate.save_sync("tts-output.mp3")

        output = self.rvc_model.vc_single(
            0,
            "C:/Users/icema/Music/VO_Clorinde_About_Charlotte.mp3",
            2,
            0,
            "rmvpe",
            "tts/models/kokomi/kokomi-rvc2.index",
            None,
            0,
            0.88,
            0.5,
            0.5,
            0.33
        )

        write("rvc-output.wav", output[1][0], output[1][1])

        #sd.play(output[1][1], output[1][0])
        #sd.wait()

    def tts_directory(self, path: str, output_path: str):
        for i, fn in enumerate(os.listdir(path)):
            output = self.rvc_model.vc_single(
                0,
                f"{path}/{fn}",
                2,
                0,
                "rmvpe",
                "tts/models/kokomi/kokomi-rvc2.index",
                None,
                0,
                0.88,
                0.66,
                0.5,
                0.1
            )

            write(f"{output_path}/{i}.wav", output[1][0], output[1][1])
