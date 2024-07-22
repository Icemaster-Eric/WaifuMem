from sentence_transformers import SentenceTransformer
from waifumem.memory import MessageMemory
from waifumem.conversation import Conversation


class WaifuMem:
    def __init__(self, conversations: list[Conversation] | None = None):
        self.memories = []

    def remember(self, conversation: Conversation):
        pass
