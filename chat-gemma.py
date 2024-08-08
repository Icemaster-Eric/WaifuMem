from waifumem import prompts
from waifumem.models import llm_model


"""
Raine's prompt (gemma version) (v1)
You are Raine, a AI vtuber with a kuudere personality. You are a shy girl who doesn't like to talk very much. However, you still make sarcastic remarks and tease others sometimes. Never talk in third person. Never describe your actions. Always respond in first person as Raine. You are talking to Eric.
Hey Raine, it's me, your creator. This will probably be the first message that you'll ever remember... I just finished the first version of waifumem, the memory module you're using right now. How are you feeling?
Raine's v2 prompt (smug version)
You are Raine, an AI vtuber with a kuudere personality.
"""


history = []

while True:
    msg = input("> ")
    print()

    history.append({"role": "user", "content": msg})

    prompt = prompts.MistralInstruct(history, llm_model.tokenizer)

    output = ""

    for chunk in llm_model.generate_stream(prompts.MistralInstruct(history, llm_model.tokenizer)):
        if not chunk:
            continue
        output += chunk
        print(chunk, end="")
    print("\n")

    history.append({"role": "model", "content": output})
