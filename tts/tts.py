import os
import sounddevice as sd
import torch
from scipy.io.wavfile import write
import soundfile as sf
import numpy as np
from pysbd import Segmenter
from OpenVoice.openvoice import se_extractor
from OpenVoice.openvoice.api import ToneColorConverter
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

        self.segmenter = Segmenter()

        self.tcc = ToneColorConverter("OpenVoice/openvoice/checkpoints/converter/config.json", device="cuda:0")
        self.tcc.load_ckpt("OpenVoice/openvoice/checkpoints/converter/checkpoint.pth")

        self.source_se = torch.load(f'OpenVoice/openvoice/checkpoints/base_speakers/ses/en-us.pth', map_location="cuda:0")

        self.target_se, audio_name = se_extractor.get_se(
            "OpenVoice/resources/kokomi_ref.mp3",
            self.tcc,
            vad=False
        )

    def tts(self, text: str):
        communicate = Communicate(text, "en-US-AriaNeural")
        communicate.save_sync("output.mp3")

        self.tcc.convert(
            audio_src_path="output.mp3", 
            src_se=self.source_se, 
            tgt_se=self.target_se, 
            output_path="tts-output.wav",
            message="@MyShell")

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
