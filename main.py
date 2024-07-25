import json
from waifumem import WaifuMem, Conversation


def main():
    with open("waifumem/data/messages.txt", "r", encoding="utf-8") as f:
        messages = [
            {
                "message": m[m.find("]: ") + 3:],
                "user": m[1:m.find("]: ")],
                "timestamp": int(m.split()[-1]),
            } for m in f.readlines()
        ]

    conversations = [Conversation(m) for m in zip(*[iter(messages)]*20)]

    waifumem = WaifuMem(conversations)

    with open("summaries.json", "w") as f:
        json.dump(waifumem.summaries, f)
    with open("topics.json", "w") as f:
        json.dump(waifumem.topics, f)

    print(waifumem.search("Best healer"))

    waifumem.save("mem.xz")


if __name__ == "__main__":
    main()
