from llama_cpp import Llama


llm = Llama(
    model_path="waifumem/models/gemma-2-27b-it-Q5_K_L.gguf",
    chat_format="gemma",
    n_gpu_layers=-1
)


def summarize(text: str) -> str:
    """Summarizes a conversation

    Args:
        text (str): Conversation as a string

    Returns:
        str: Summary
    """
    return llm.create_chat_completion([
        {"role": "user", "content": f"You are a smart AI that summarizes conversations efficiently. Summarize the following conversation:\n{text}"}
    ])["choices"][0]["message"]["content"]
