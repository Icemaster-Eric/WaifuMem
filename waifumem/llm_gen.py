from llama_cpp import Llama
llm = Llama(
      model_path="waifumem/models/gemma-2-27b-it-Q5_K_L.gguf",
      chat_format="gemma",
      n_gpu_layers=-1
)
for token in llm.create_chat_completion(
    messages = [
        {"role": "user", "content": """You are a smart AI that summarizes the most relevant topics of a conversation. Return only the most important, relevant topics in the following conversation in a JSON list:
Eric: hey
Josh: yo
Eric: how's it going?
Josh: the craziest thing happened today, actually
Josh: I actually got a gf
Eric: WHAT
Charlette: THERE'S NO WAY
Josh: yep
Eric: bro spill the beans rn
Josh: ok so basically in my ESL class
Josh: there's this really cute russian girl
Josh: and we somehow ended up playing fireboy and watergirl together
Eric: :skull:
Charlette: bro are you 5 or something
Josh: shut it
Josh: not like any of you even talk to girls
HydrogenMacro: girls? what are those?"""},
    ], temperature=0.4, stream=True
):
    if token["choices"][0]["delta"].get("content"):
        print(token["choices"][0]["delta"].get("content"), end="")

print()
