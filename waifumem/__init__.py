from sentence_transformers import SentenceTransformer
from waifumem.memory import Memory
from waifumem.conversation import Conversation


class WaifuMem:
    def __init__(self, conversations: list[Conversation] | None = None):
        self.memories = []

        for conversation in conversations:
            for message in conversation.messages:
                # calculate whether the message is fit to become a memory or not (importance)
                # I should probably use a finetuned distilbert model for this
                # 0 = unfit, 1 = slightly important, 2 = important
                importance = 1
                self.memories.append(Memory(message, importance))

    def remember(self, conversation: Conversation):
        pass
