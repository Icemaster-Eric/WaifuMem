import json
"""from waifumem import WaifuMem, Conversation


def main():
    with open("waifumem/data/messages.txt", "r", encoding="utf-8") as f:
        messages = [
            {
                "message": m[m.find("]: ") + 3:],
                "user": m[1:m.find("]: ")],
                "timestamp": int(m.split()[-1]),
            } for m in f.readlines()
        ]

    with open("summaries.json", "r") as f:
        summaries = json.load(f)
    with open("topics.json", "r") as f:
        topics = json.load(f)

    conversations = [Conversation(m, summary=summaries[i], topics=topics[i]) for i, m in enumerate(zip(*[iter(messages)]*20))]

    waifumem = WaifuMem(conversations)

    print(waifumem.search("How much money does Zhongli have?", top_k=2))

    #waifumem.save("mem.xz")


def test():
    waifumem = WaifuMem.load("mem.xz")

    print(waifumem.conversations[0].messages)

    while True:
        query = input("|> ")

        print(waifumem.search(query, top_k=3))"""

from tts import RaineTTS


def tts_test():
    raine_tts = RaineTTS()

    raine_tts.tts("""We're no strangers to love.
You know the rules and so do I.
A full commitment's what I'm thinking of.
You wouldn't get this from any other guy.""")


def edge_tts_test():
    import edge_tts

    communicate = edge_tts.Communicate("Wanna come over for tea?", "en-US-AriaNeural")

    for chunk in communicate.stream_sync():
        if chunk["type"] == "WordBoundary":
            pass


if __name__ == "__main__":
    tts_test()
