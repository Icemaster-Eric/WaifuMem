from typing import Literal
import pickle
import lzma
from tqdm import tqdm
from torch import set_default_device; set_default_device("cuda")
from sentence_transformers.util import semantic_search
from models import llm_model, embedding_model
from waifumem.conversation import Conversation


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


class Knowledge: # stores knowledge of the model? unsure rn
    def __init__(self):
        pass


class WaifuMem:
    def __init__(self, conversations: list[Conversation] = []):
        self.conversations = conversations
        self.summaries = []
        self.summary_embeddings = []
        self.topics = []
        self.topic_embeddings = []

        for conversation in tqdm(conversations, desc="Generating memory"):
            self.remember(conversation)

    def remember(self, conversation: Conversation):
        # summarize conversation
        summary = get_summary(conversation.get_text())
        self.summaries.append(summary)
        # embed conversation
        self.summary_embeddings.append(embedding_model.encode(summary, convert_to_tensor=True))

        # get topics of conversation
        topics = get_topics(conversation.get_text())
        self.topics.append(topics)
        # embed conversation
        self.topic_embeddings.append(embedding_model.encode(topics, convert_to_tensor=True))

    def search_conversation(self, message_embedding, conversation_id: str) -> list[dict[Literal["message", "user", "timestamp"], str | float]]:
        """Semantically searches for relevant messages in a Conversation

        Args:
        conversation_id (str): `Conversation.id`

        Returns:
            list: messages from `Conversation.messages`
        """
        conversation = next(conv for conv in self.conversations if conv.id == conversation_id)

        results = semantic_search(message_embedding, conversation.message_embeddings)[0]

        return [conversation.messages[result["corpus_id"]] for result in results]

    def search_messages(self):
        # search all messages in all conversations
        pass

    def search_knowledge(self):
        pass

    def search(self, text: str):
        query = embedding_model.encode(text, prompt_name="query")

        # find relevant conversations by summary (adjust to similarity search based on current conversation's summary?)
        summary_results = semantic_search(query, self.summary_embeddings)[0]
        # find relevant conversations by topics (adjust to similarity search based on current conversation's topics?)
        topic_results = semantic_search(query, self.topic_embeddings)[0]

        # create list of conversations
        for i in [result["corpus_id"] for result in summary_results + topic_results]:
            conversation = self.conversations[i]

            yield self.search_conversation(query, conversation.id)

    def save(self, path: str):
        with lzma.open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "WaifuMem":
        with lzma.open(path, "rb") as f:
            waifumem = pickle.load(f)

        return waifumem
