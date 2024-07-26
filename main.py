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

    with open("summaries.json", "r") as f:
        summaries = json.load(f)
    with open("topics.json", "r") as f:
        topics = json.load(f)

    conversations = [Conversation(m, summary=summaries[i], topics=topics[i]) for i, m in enumerate(zip(*[iter(messages)]*20))]

    waifumem = WaifuMem(conversations)

    print(waifumem.search("How much money does Zhongli have?", top_k=10))

    waifumem.save("mem.xz")


def test():
    waifumem = WaifuMem.load("mem.xz")

    print(waifumem.conversations[0].messages)

    while True:
        query = input("|> ")

        print(waifumem.search(query))


if __name__ == "__main__":
    test()
