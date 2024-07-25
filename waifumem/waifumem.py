import pickle
import lzma
from tqdm import tqdm
from numpy import concatenate
#from torch import set_default_device; set_default_device("cuda")
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search
from waifumem.memory import Memory
from waifumem.conversation import Conversation


llm_model = Llama(
    model_path="waifumem/models/gemma-2-27b-it-Q5_K_L.gguf",
    chat_format="gemma",
    n_ctx=8192,
    n_gpu_layers=-1,
    use_mmap=False,
    verbose=False
)
embedding_model = SentenceTransformer("Snowflake/snowflake-arctic-embed-l")


def get_summary(text: str) -> str:
    """Summarizes a conversation

    Args:
        text (str): Conversation as a string

    Returns:
        str: Summary
    """
    return llm_model.create_chat_completion([
        {"role": "user", "content": f"You are a smart AI that summarizes conversations. Summarize the following conversation as briefly as possible, without mentioning anything unnecessary:\n{text}"}
    ], temperature=0.3, stop="\n")["choices"][0]["message"]["content"]


def get_topics(text: str) -> str:
    """Returns the topics of a conversation in natural language. May switch to using lmformatenforcer in the future.

    Args:
        text (str): Conversation as a string

    Returns:
        str: Topics
    """
    return llm_model.create_chat_completion([
        {"role": "user", "content": f"You are a smart AI that finds the relevant topics of conversations. Return the topics of the conversation as briefly as possible, without mentioning anything unnecessary besides the topics:\n{text}"}
    ], temperature=0.3, stop="\n")["choices"][0]["message"]["content"]


class WaifuMem:
    def __init__(self, conversations: list[Conversation] = []):
        self.conversations = conversations
        self.memories = []
        self.summaries = []
        self.summary_embeddings = None
        self.topics = []
        self.topic_embeddings = None

        for conversation in tqdm(conversations, desc="Generating memory"):
            self.remember(conversation)

    def remember(self, conversation: Conversation):
        for message in conversation.messages:
            # calculate whether the message is fit to become a memory or not (importance)
            # I should probably use a finetuned distilbert model for this
            # 0 = unfit, 1 = slightly important, 2 = important
            importance = 1
            self.memories.append(Memory(message, importance))

        # summarize conversation
        summary = get_summary(conversation.get_text())
        self.summaries.append(summary)
        # embed conversation
        summary_embeddings = embedding_model.encode([summary])

        if self.summary_embeddings is None:
            self.summary_embeddings = summary_embeddings
        else:
            self.summary_embeddings = concatenate([self.summary_embeddings, summary_embeddings])

        # get topics of conversation
        topics = get_topics(conversation.get_text())
        self.topics.append(topics)
        # embed conversation
        topic_embeddings = embedding_model.encode([topics])

        if self.topic_embeddings is None:
            self.topic_embeddings = topic_embeddings
        else:
            self.topic_embeddings = concatenate([self.summary_embeddings, topic_embeddings])

    def search(self, text: str):
        query = embedding_model.encode(text, prompt_name="query")

        return semantic_search(query, self.summary_embeddings)

    def save(self, path: str):
        with lzma.open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "WaifuMem":
        with lzma.open(path, "rb") as f:
            waifumem = pickle.load(f)

        return waifumem
