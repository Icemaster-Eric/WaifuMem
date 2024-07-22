class Memory:
    def __init__(self, message: dict, importance: int):
        self.text = message["message"]
        self.user = message["user"]
        self.timestamp = message["timestamp"]
        self.importance = importance
