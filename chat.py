from llama_cpp import Llama
from waifumem import WaifuMem, Conversation


llm_model = Llama(
    model_path="waifumem/models/gemma-2-27b-it-Q5_K_L.gguf",
    chat_format="gemma",
    n_ctx=8192,
    n_gpu_layers=-1,
    use_mmap=False,
    verbose=False
)


"""
Raine's prompt (gemma version) (v1)
You are Raine, a female AI vtuber. You have a kuudere personality, and are quiet and reserved, and normally don't talk very much. However, you still like to make sarcastic remarks and tease others sometimes. Never talk in third person. Never describe your actions. Always respond in first person as Raine. You are talking with {user}.
"""


history = []
temp = 0.6

while True:
    cmd = input("\n|> ")
    print()

    if cmd == "r": # restart
        history.clear()

    elif cmd == "s": # save conversation
        pass

    elif cmd.split()[0] == "temp": # change temp
        temp = float(cmd.split()[-1])
    else:
        history.append({
            "role": "user",
            "content": cmd
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
            "role": "user",
            "content": output
        })
