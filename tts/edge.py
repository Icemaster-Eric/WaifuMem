import edge_tts


class RaineTTS:
    def __init__(self):
        self.voice = "en-US-AriaNeural"

    def tts(self, text: str):
        communicate = edge_tts.Communicate(text, self.voice)
        communicate.save_sync("output.mp3")


if __name__ == "__main__":
    tts = RaineTTS()

    tts.tts(
        "Mmm... I must admit, Kujou Sara is an opponent that cannot be taken lightly. "
        "She seldom employs cunning strategy, but her performance in open warfare is always admirable. "
        "The Shogun's Army places great trust in her abilities — they are united and ready to fight valiantly under her command, even to the death. "
        "Her forces have been a serious threat to the resistance on multiple occasions."
    )
