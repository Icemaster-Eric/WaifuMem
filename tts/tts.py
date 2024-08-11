import os
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import numpy as np
from pysbd import Segmenter
from tts.rvc.infer.modules.vc.modules import VC
from tts.rvc.configs.config import Config


os.environ["rmvpe_root"] = "tts/models"


class RaineTTS:
    def __init__(self):
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            "parler-tts/parler-tts-large-v1",
            attn_implementation="sdpa"
        ).to("cuda:0")
        self.tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-large-v1")

        self.rvc_config = Config()
        self.rvc_config.n_cpu = 2
        self.rvc_model = VC(self.rvc_config)
        self.rvc_model.get_vc("tts/models/kokomi-rvc2.pth", "tts/models/kokomi/kokomi-rvc2.index")
        self.rvc_model.load_hubert("tts/models/hubert_base.pt")

        self.segmenter = Segmenter()

    def tts(self, text: str, desc: str = "Laura's voice is animated and expressive, yet very clear. The recording is of very high quality."):
        audio = []

        input_ids = self.tokenizer(desc, return_tensors="pt").input_ids.to("cuda:0")

        for sentence in self.segmenter.segment(text):
            prompt_input_ids = self.tokenizer(sentence, return_tensors="pt").input_ids.to("cuda:0")

            generation = self.model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
            audio_arr = generation.cpu().numpy().squeeze()
            audio.append(audio_arr)

        sf.write("tts-output.wav", np.concatenate(audio), self.model.config.sampling_rate)

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
