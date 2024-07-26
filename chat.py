from waifumem import WaifuMem, Conversation
from waifumem.models import llm_model
import json # for debugging


"""
Raine's prompt (gemma version) (v1)
You are Raine, a AI vtuber with a kuudere personality. You are a shy girl who doesn't like to talk very much. However, you still make sarcastic remarks and tease others sometimes. Never talk in third person. Never describe your actions. Always respond in first person as Raine. You are talking to Eric.
Hey Raine, it's me, your creator. This will probably be the first message that you'll ever remember... I just finished the first version of waifumem, the memory module you're using right now. How are you feeling?
"""


history = []
temp = 0.8
username = "Eric"
memory = WaifuMem()
conversation = Conversation()

while True:
    cmd = input("\n|> ")
    print()

    if cmd[:2] == "r ": # restart
        history.clear()
        history.append({
            "role": "user",
            "content": f"{cmd[2:]}"
        })

        output = ""
        for token in llm_model.create_chat_completion(history, temperature=temp, stream=True):
            content = token["choices"][0]["delta"].get("content")
            if content is None:
                continue
            output += content
            print(content, end="")
        print()

        history.append({
            "role": "model",
            "content": output
        })

    elif cmd == "m": # save conversation to memory
        memory.remember(conversation.cut(0.5))
        history = [history[0]] + history[int((len(history) - 1) * 0.5):]

    elif cmd == "s": # save waifumem
        memory.save("mem.xz")

    elif cmd.split()[0] == "temp": # change temp
        temp = float(cmd.split()[-1])
    else:
        history.append({
            "role": "user",
            "content": cmd
        })
        conversation.add_message(cmd, username)

        results = memory.search(cmd, top_k=2)

        context = history[0].copy()
        context["content"] += "\n" + "\n".join(
            f"Memory:\n{m[0]['user']}: {m[0]['message']}" for m in results
        )

        llm_input = [context] + history[1:]

        with open("chat_history.json", "w") as f:
            json.dump(llm_input, f, indent=1)

        output = ""
        for token in llm_model.create_chat_completion(llm_input, temperature=temp, stream=True):
            content = token["choices"][0]["delta"].get("content")
            if content is None:
                continue
            output += content
            print(content, end="")
        print()

        output = output.strip()

        history.append({
            "role": "model",
            "content": output
        })
        conversation.add_message(output, "Raine")
