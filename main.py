from waifumem import WaifuMem, Conversation


def main():
    with open("waifumem/data/messages.txt", "r", encoding="utf-8") as f:
        messages = [
            {
                "message": m[m.find("]: ") + 3:],
                "user": m[1:m.find("]: ")],
                "timestamp": int(m.split()[-1]),
            } for m in f.readlines()
        ][:60]
    
    conversations = [Conversation(m) for m in zip(*[iter(messages)]*20)]

    waifumem = WaifuMem(conversations)

    print(waifumem.summaries)

    print(waifumem.search("Best nahida teams"))


if __name__ == "__main__":
    main()
