from styletts2 import tts
from scipy.io.wavfile import write
from tts.rvc.misc import load_hubert, get_vc, vc_single
import sounddevice as sd
import soundfile as sf
import re
import time


class RaineTTS:
    def __init__(self):
        self.tts_model = tts.StyleTTS2()

    def tts(self, text: str):
        pattern = '|'.join('(?<={})'.format(re.escape(delim)) for delim in (". ", "? ", "! "))

        for sentence in re.split(pattern, text):
            print(sentence)

            self.tts_model.inference(sentence, output_wav_file="output.wav")

            get_vc("kokomi-rvc2.pth", "tts/models/kokomi", 0.33, 0.5)

            hubert_model = load_hubert("tts/models/hubert_base.pt")

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
            print("done!")

            write("output.wav", wav_opt[1][0], wav_opt[1][1])

            sd.play(*sf.read("output.wav"))
            sd.wait()
            time.sleep(2)
