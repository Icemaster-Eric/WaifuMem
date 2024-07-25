from waifumem.conversation import Conversation


class Memory:
    def __init__(self, message: dict, conversation: Conversation, importance: int):
        self.text = message["message"]
        self.user = message["user"]
        self.timestamp = message["timestamp"]
        self.conversation_id = conversation.id
        self.importance = importance
