from multiprocessing import Queue
from styletts2 import tts
import torch
import torch.nn.functional as F
import torchaudio.transforms as tat
import numpy as np
import librosa
from tts.rvc.torchgate import TorchGate
from tts.rvc.rtrvc import RVC
from tts.rvc.configs.config import Config


class GUIConfig:
    def __init__(self) -> None:
        self.pth_path: str = "tts/models/kokomi/kokomi-rvc2.pth"
        self.index_path: str = "tts/models/kokomi/kokomi-rvc2.index"
        self.pitch: int = 3
        self.samplerate: int = 40000
        self.block_time: float = 1.0  # s
        self.buffer_num: int = 1
        self.threhold: int = -60
        self.crossfade_time: float = 0.05
        self.extra_time: float = 2.5
        self.I_noise_reduce = False
        self.O_noise_reduce = False
        self.rms_mix_rate = 0.0
        self.index_rate = 0.3
        self.f0method = "rmvpe"
        self.sg_input_device = ""
        self.sg_output_device = ""


class RaineTTS:
    def __init__(self):
        self.tts_model = tts.StyleTTS2()

        # rvc stuff
        self.config = Config()
        self.in_q = Queue()
        self.out_q = Queue()

        self.gui_config = GUIConfig()
        self.config = None  # Initialize Config object as None
        self.flag_vc = False
        self.function = "vc"
        self.delay_time = 0
        self.rvc = None  # Initialize RVC object as None

    def start_vc(self):
        torch.cuda.empty_cache()
        self.flag_vc = True
        self.rvc = RVC(
            self.gui_config.pitch,
            self.gui_config.pth_path,
            self.gui_config.index_path,
            self.gui_config.index_rate,
            0,
            0,
            0,
            self.config,
            self.rvc if self.rvc else None,
        )
        self.gui_config.samplerate = self.rvc.tgt_sr
        self.zc = self.rvc.tgt_sr // 100
        self.block_frame = (
            int(
                np.round(
                    self.gui_config.block_time
                    * self.gui_config.samplerate
                    / self.zc
                )
            )
            * self.zc
        )
        self.block_frame_16k = 160 * self.block_frame // self.zc
        self.crossfade_frame = (
            int(
                np.round(
                    self.gui_config.crossfade_time
                    * self.gui_config.samplerate
                    / self.zc
                )
            )
            * self.zc
        )
        self.sola_search_frame = self.zc
        self.extra_frame = (
            int(
                np.round(
                    self.gui_config.extra_time
                    * self.gui_config.samplerate
                    / self.zc
                )
            )
            * self.zc
        )
        self.input_wav = torch.zeros(
            self.extra_frame + self.crossfade_frame + self.sola_search_frame + self.block_frame,
            device=self.config.device,
            dtype=torch.float32,
        )
        self.input_wav_res = torch.zeros(
            160 * self.input_wav.shape[0] // self.zc,
            device=self.config.device,
            dtype=torch.float32,
        )
        self.pitch = np.zeros(self.input_wav.shape[0] // self.zc, dtype="int32")
        self.pitchf = np.zeros(self.input_wav.shape[0] // self.zc, dtype="float64")
        self.sola_buffer = torch.zeros(self.crossfade_frame, device=self.config.device, dtype=torch.float32)
        self.nr_buffer = self.sola_buffer.clone()
        self.output_buffer = self.input_wav.clone()
        self.res_buffer = torch.zeros(2 * self.zc, device=self.config.device, dtype=torch.float32)
        self.valid_rate = 1 - (self.extra_frame - 1) / self.input_wav.shape[0]
        self.fade_in_window = (
            torch.sin(0.5 * np.pi * torch.linspace(0.0, 1.0, steps=self.crossfade_frame, device=self.config.device, dtype=torch.float32)) ** 2
        )
        self.fade_out_window = 1 - self.fade_in_window
        self.resampler = tat.Resample(
            orig_freq=self.gui_config.samplerate,
            new_freq=16000,
            dtype=torch.float32,
        ).to(self.config.device)
        self.tg = TorchGate(
            sr=self.gui_config.samplerate, n_fft=4 * self.zc, prop_decrease=0.9
        ).to(self.config.device)

    def audio_callback(self, indata: np.ndarray, outdata: np.ndarray):
        indata = librosa.to_mono(indata.T)

        if self.gui_config.threhold > -60:
            rms = librosa.feature.rms(y=indata, frame_length=4 * self.zc, hop_length=self.zc)
            db_threhold = (librosa.amplitude_to_db(rms, ref=1.0)[0] < self.gui_config.threhold)
            for i in range(db_threhold.shape[0]):
                if db_threhold[i]:
                    indata[i * self.zc : (i + 1) * self.zc] = 0

        self.input_wav[: -self.block_frame] = self.input_wav[self.block_frame :].clone()
        self.input_wav[-self.block_frame :] = torch.from_numpy(indata).to(self.config.device)
        self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[self.block_frame_16k :].clone()

        if self.gui_config.I_noise_reduce and self.function == "vc":
            input_wav = self.input_wav[-self.crossfade_frame - self.block_frame - 2 * self.zc :]
            input_wav = self.tg(input_wav.unsqueeze(0), self.input_wav.unsqueeze(0))[0, 2 * self.zc :]
            input_wav[: self.crossfade_frame] *= self.fade_in_window
            input_wav[: self.crossfade_frame] += self.nr_buffer * self.fade_out_window
            self.nr_buffer[:] = input_wav[-self.crossfade_frame :]
            input_wav = torch.cat((self.res_buffer[:], input_wav[: self.block_frame]))
            self.res_buffer[:] = input_wav[-2 * self.zc :]
            self.input_wav_res[-self.block_frame_16k - 160 :] = self.resampler(input_wav)[160:]
        else:
            self.input_wav_res[-self.block_frame_16k - 160 :] = self.resampler(self.input_wav[-self.block_frame - 2 * self.zc :])[160:]

        if self.function == "vc":
            f0_extractor_frame = self.block_frame_16k + 800
            if self.gui_config.f0method == "rmvpe":
                f0_extractor_frame = (5120 * ((f0_extractor_frame - 1) // 5120 + 1) - 160)
            infer_wav = self.rvc.infer(
                self.input_wav_res,
                self.input_wav_res[-f0_extractor_frame:].cpu().numpy(),
                self.block_frame_16k,
                self.valid_rate,
                self.pitch,
                self.pitchf,
                self.gui_config.f0method,
            )
            infer_wav = infer_wav[-self.crossfade_frame - self.sola_search_frame - self.block_frame :]
        else:
            infer_wav = self.input_wav[-self.crossfade_frame - self.sola_search_frame - self.block_frame :].clone()

        if (self.gui_config.O_noise_reduce and self.function == "vc") or (self.gui_config.I_noise_reduce and self.function == "im"):
            self.output_buffer[: -self.block_frame] = self.output_buffer[self.block_frame :].clone()
            self.output_buffer[-self.block_frame :] = infer_wav[-self.block_frame :]
            infer_wav = self.tg(infer_wav.unsqueeze(0), self.output_buffer.unsqueeze(0)).squeeze(0)

        if self.gui_config.rms_mix_rate < 1 and self.function == "vc":
            rms1 = librosa.feature.rms(y=self.input_wav_res[-160 * infer_wav.shape[0] // self.zc :].cpu().numpy(), frame_length=640, hop_length=160)
            rms1 = torch.from_numpy(rms1).to(self.config.device)
            rms1 = F.interpolate(rms1.unsqueeze(0), size=infer_wav.shape[0] + 1, mode="linear", align_corners=True)[0, 0, :-1]
            rms2 = librosa.feature.rms(y=infer_wav[:].cpu().numpy(), frame_length=4 * self.zc, hop_length=self.zc)
            rms2 = torch.from_numpy(rms2).to(self.config.device)
            rms2 = F.interpolate(rms2.unsqueeze(0), size=infer_wav.shape[0] + 1, mode="linear", align_corners=True)[0, 0, :-1]
            rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-3)
            infer_wav *= torch.pow(rms1 / rms2, torch.tensor(1 - self.gui_config.rms_mix_rate))

        conv_input = infer_wav[None, None, : self.crossfade_frame + self.sola_search_frame]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(F.conv1d(conv_input**2, torch.ones(1, 1, self.crossfade_frame, device=self.config.device)) + 1e-8)

        sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])

        infer_wav = infer_wav[sola_offset : sola_offset + self.block_frame + self.crossfade_frame]
        infer_wav[: self.crossfade_frame] *= self.fade_in_window
        infer_wav[: self.crossfade_frame] += self.sola_buffer * self.fade_out_window
        self.sola_buffer[:] = infer_wav[-self.crossfade_frame :]

        outdata[:] = infer_wav[: -self.crossfade_frame].repeat(2, 1).t().cpu().numpy()

    def tts(self, text: str):
        self.tts_model.inference(text, output_wav_file="output.wav")

        #self.audio_callback(...)
