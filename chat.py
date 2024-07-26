from waifumem import WaifuMem, Conversation
from waifumem.models import llm_model


"""
Raine's prompt (gemma version) (v1)
You are Raine, a AI vtuber with a kuudere personality. You are a shy girl who doesn't like to talk very much. However, you still make sarcastic remarks and tease others sometimes. Never talk in third person. Never describe your actions. Always respond in first person as Raine.
"""


history = []
temp = 0.6
username = "Youkai"
memory = WaifuMem()
conversation = Conversation()

while True:
    cmd = input("\n|> ")
    print()

    if cmd[:2] == "r ": # restart
        history.clear()
        history.append({
            "role": "user",
            "content": f"{username}: {cmd[2:]}"
        })

    elif cmd == "m": # save conversation to memory
        memory.remember(conversation.cut(1))
        # reset afterwards because I'm too lazy to make it actually work rn

    elif cmd == "s": # save waifumem
        memory.save("mem.xz")

    elif cmd.split()[0] == "temp": # change temp
        temp = float(cmd.split()[-1])
    else:
        history.append({
            "role": "user",
            "content": f"{username}: {cmd}"
        })
        conversation.add_message(cmd, username)
        output = ""
        for token in llm_model.create_chat_completion(history, temperature=temp, stream=True):
            content = token["choices"][0]["delta"].get("content")
            if content is None:
                continue
            output += content
            print(content, end="")
        print()

        if not output.lower().startswith("raine"):
            output = f"Raine: {output}"

        history.append({
            "role": "model",
            "content": output
        })
        conversation.add_message(output, "Raine")
