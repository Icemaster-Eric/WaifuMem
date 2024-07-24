from sentence_transformers import SentenceTransformer
from waifumem.memory import Memory
from waifumem.conversation import Conversation
from waifumem.llm import summarize


class WaifuMem:
    def __init__(self, conversations: list[Conversation] = []):
        self.memories = []

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
        # embed conversation

        # find similar conversation topics
