from styletts2 import tts


my_tts = tts.StyleTTS2()

out = my_tts.inference(
    "You're putting a lot of effort into this. Make me look cool!",
    target_voice_path="tts/ref.wav",
    alpha=0.3,
    beta=0.7,
    output_wav_file="output.wav"
)
