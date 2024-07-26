from typing import Literal
import time
from uuid import uuid4
from copy import deepcopy
from waifumem.models import embedding_model


class Conversation:
    def __init__(self, messages: list[dict[Literal["message", "user", "timestamp"], str | float]] = [], summary: str | None = None, topics: str | None = None):
        """Creates a conversation object that is mutable

        Args:
            messages (list[dict] | None, optional): List of message objects with message (str, should be in the format `{user}: {message}`), user (str) and timestamp (int, epoch seconds) keys. Defaults to None.
        """
        self.id = uuid4().hex
        self.messages = messages
        self.summary = summary
        self.topics = topics

        # cluster messages
        for i, message in enumerate(self.messages):
            if i == 0:
                continue

            if message["user"] == self.messages[i - 1]["user"]:
                if message["timestamp"] - self.messages[i - 1]["timestamp"] > 120:
                    continue

                self.messages[i - 1]["message"] += "\n" + message["message"]
                self.messages[i - 1]["timestamp"] = message["timestamp"]
                self.messages.remove(message)

        if self.messages:
            self.message_embeddings = embedding_model.encode([
                f"{message['user']}: {message['message']}" for message in self.messages # convert timestamp to human-readable format and include here later (?)
            ], convert_to_tensor=True)
        else:
            self.message_embeddings = []

    def add_message(self, message: str, user: str, timestamp: float | None = None):
        # cluster message
        if self.messages:
            if user == self.messages[-1]["user"]:
                if timestamp - self.messages[-1]["timestamp"] > 120:
                    self.messages[-1] += "\n" + message
                    self.messages[-1]["timestamp"] = timestamp
                    return

        self.messages.append({
            "message": message,
            "user": user,
            "timestamp": timestamp or time.time()
        })

        self.message_embeddings.append(embedding_model.encode(self.messages[-1]["message"], convert_to_tensor=True))

    def cut(self, ratio: float = 0.5) -> "Conversation":
        """Cuts the `.messages` list by the ratio and returns a Conversation object with the former slice of the `.messages` list.

        Args:
            ratio (float, optional): The ratio to cut the `.messages` list. Defaults to 0.5.

        Returns:
            Conversation: A new copy of the object with the former slice of the `.messages` list.
        """
        conversation = deepcopy(self)
        split = int(len(self.messages) * ratio)
        conversation.messages = self.messages[:split]
        self.messages = self.messages[split:]

        return conversation

    def get_text(self):
        return "\n".join(f"{message['message']}" for message in self.messages)
