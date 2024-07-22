from llama_cpp import Llama
llm = Llama(
      model_path="waifumem/models/gemma-2-27b-it-Q5_K_L.gguf",
      chat_format="gemma"
)
print(llm.create_chat_completion(
    messages = [
        {"role": "system", "content": "You are a smart AI that summarizes conversations in a succinct and efficient manner. Summarize the following conversation."},
        {"role": "User A", "content": "hey"},
        {"role": "User B", "content": "yo"},
        {"role": "User A", "content": "how's it going?"},
        {"role": "User B", "content": "the craziest thing happened today actually"},
        {"role": "User B", "content": "I actually got a gf"},
        {"role": "User A", "content": "WHAT"},
        {"role": "User C", "content": "THERE'S NO WAY"},
        {"role": "User A", "content": "bro spill the beans rn"},
        {"role": "User B", "content": "ok so basically in my ESL class"},
        {"role": "User B", "content": "there's this really cute russian girl"},
        {"role": "User B", "content": "and we somehow ended up talking and playing games together"},
        {"role": "User A", "content": ":skull:"},
        {"role": "User C", "content": "that doesn't make you boyfriend and girlfriend"},
        {"role": "User B", "content": "I asked her out"},
        {"role": "User B", "content": "and she agreed"},
        {"role": "User A", "content": "unfriending you rn"},
        {"role": "User C", "content": "nah bro you ain't my friend anymore"},
    ]
))
