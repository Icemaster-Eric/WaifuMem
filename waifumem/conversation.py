from __future__ import annotations
import time
from uuid import uuid4
from copy import deepcopy


class Conversation:
    def __init__(self, messages: list[dict] | None = None):
        self.id = uuid4().hex
        self.messages = messages or []

    def add_message(self, message: str, user: str, timestamp: float | None):
        self.messages.append({
            "message": message,
            "user": user,
            "timestamp": timestamp or time.time()
        })

    def cut(self, ratio: float = 0.5) -> Conversation:
        """Cuts the `.messages` list by the ratio and saves the latter slice while returning a Conversation object with the former slice of the `.messages` list.

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
