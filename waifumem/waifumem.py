from tqdm import tqdm
from numpy import concatenate
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search
from waifumem.memory import Memory
from waifumem.conversation import Conversation


"""llm_model = Llama(
    model_path="waifumem/models/gemma-2-27b-it-Q5_K_L.gguf",
    chat_format="gemma",
    n_ctx=8192,
    n_gpu_layers=-1,
    use_mmap=False,
    verbose=False
)"""
embedding_model = SentenceTransformer("Snowflake/snowflake-arctic-embed-l")


def summarize(text: str) -> str:
    """Summarizes a conversation

    Args:
        text (str): Conversation as a string

    Returns:
        str: Summary
    """
    return text
    return llm_model.create_chat_completion([
        {"role": "user", "content": f"You are a smart AI that summarizes conversations efficiently. Summarize the following conversation:\n{text}"}
    ], temperature=0.3)["choices"][0]["message"]["content"]


class WaifuMem:
    def __init__(self, conversations: list[Conversation] = []):
        self.memories = []
        self.summaries = []
        self.summary_embeddings = None

        for conversation in tqdm(conversations):
            self.remember(conversation)

    def remember(self, conversation: Conversation):
        for message in conversation.messages:
            # calculate whether the message is fit to become a memory or not (importance)
            # I should probably use a finetuned distilbert model for this
            # 0 = unfit, 1 = slightly important, 2 = important
            importance = 1
            self.memories.append(Memory(message, importance))

        # summarize conversation
        summary = summarize(conversation.get_text())
        self.summaries.append(summary)
        # embed conversation
        embeddings = embedding_model.encode([summary])

        if self.summary_embeddings is None:
            self.summary_embeddings = embeddings
        else:
            self.summary_embeddings = concatenate([self.summary_embeddings, embeddings])

    def search(self, text: str):
        query = embedding_model.encode(text, prompt_name="query")

        return semantic_search(query, self.summary_embeddings)
