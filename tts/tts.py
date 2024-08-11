import os
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
#from StyleTTS2.ljinference import tts_to_file
from tts.rvc.infer.modules.vc.modules import VC
from tts.rvc.configs.config import Config


os.environ["rmvpe_root"] = "tts/models"


torch_device = "cuda:0"
torch_dtype = torch.bfloat16
model_name = "parler-tts/parler-tts-large-v1"
max_length = 50


class RaineTTS:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) 
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_name,
            attn_implementation="eager"
        ).to(torch_device, dtype=torch_dtype)

        # compile the forward pass
        compile_mode = "default" # chose "reduce-overhead" for 3 to 4x speed-up
        self.model.generation_config.cache_implementation = "static"
        self.model.forward = torch.compile(self.model.forward, mode=compile_mode)

        inputs = self.tokenizer("This is for compilation", return_tensors="pt", padding="max_length", max_length=max_length).to(torch_device)

        model_kwargs = {**inputs, "prompt_input_ids": inputs.input_ids, "prompt_attention_mask": inputs.attention_mask, }

        for _ in range(1 if compile_mode == "default" else 2):
            _ = self.model.generate(**model_kwargs)

        self.rvc_config = Config()
        self.rvc_config.n_cpu = 2
        self.rvc_model = VC(self.rvc_config)
        self.rvc_model.get_vc("tts/models/kokomi-rvc2.pth", "tts/models/kokomi/kokomi-rvc2.index")
        self.rvc_model.load_hubert("tts/models/hubert_base.pt")

    def tts(self, text: str, desc: str = "Laura's voice is expressive and animated with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."):
        #tts_to_file(text, "output.wav", diffusion_steps=15, embedding_scale=2)
        input_ids = self.tokenizer(desc, return_tensors="pt").input_ids.to(torch_device)
        prompt_input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(torch_device)

        generation = self.model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        audio_arr = generation.cpu().numpy().squeeze()
        sf.write("tts-output.wav", audio_arr, self.model.config.sampling_rate)

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
