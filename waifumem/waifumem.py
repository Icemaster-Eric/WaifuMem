import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search, normalize_embeddings
from waifumem.memory import Memory
from waifumem.conversation import Conversation
from waifumem.llm import summarize

model = SentenceTransformer("Snowflake/snowflake-arctic-embed-l")


class WaifuMem:
    def __init__(self, conversations: list[Conversation] = []):
        self.memories = []
        self.summaries = []
        self.summary_embeddings = torch.Tensor()

        for conversation in conversations:
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
        embeddings = normalize_embeddings(model.encode(summary))
        self.summary_embeddings = torch.cat(self.summary_embeddings, embeddings)

    def search(self, text: str):
        query = model.encode(text, prompt_name="query")

        return semantic_search(query, self.summary_embeddings)
