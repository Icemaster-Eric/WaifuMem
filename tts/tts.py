import re
import os
import sounddevice as sd
from scipy.io.wavfile import write
from edge_tts import Communicate
from tts.rvc.infer.modules.vc.modules import VC
from tts.rvc.configs.config import Config


# I give up
os.environ["rmvpe_root"] = "tts/models"


class RaineTTS:
    def __init__(self):
        self.rvc_config = Config()
        self.rvc_config.n_cpu = 2
        self.rvc_model = VC(self.rvc_config)
        self.rvc_model.get_vc("tts/models/kokomi-rvc2.pth", "tts/models/kokomi/kokomi-rvc2.index")
        self.rvc_model.load_hubert("tts/models/hubert_base.pt")

    def tts(self, text: str):
        output = self.rvc_model.vc_single(
                0,
                "rickroll vocals.m4a",
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
        write("output.wav", output[1][0], output[1][1])
        return
        pattern = "|".join(f"(?<={re.escape(delim)})" for delim in (". ", "? ", "! "))

        for sentence in re.split(pattern, text):
            communicate = Communicate(sentence, "en-US-AriaNeural")

            subtitles: list[tuple[str, float, float]] = []

            with open("output.mp3", "wb") as f:
                for chunk in communicate.stream_sync():
                    if chunk["type"] == "audio":
                        f.write(chunk["data"])

                    elif chunk["type"] == "WordBoundary":
                        subtitles.append((
                            chunk["text"],
                            chunk["offset"] / 10000000,
                            chunk["duration"] / 10000000
                        ))

            print(subtitles)

            output = self.rvc_model.vc_single(
                0,
                "rickroll vocals.m4a",
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

            write("output.wav", output[1][0], output[1][1])

            sd.play(output[1][1], output[1][0])
            sd.wait()
