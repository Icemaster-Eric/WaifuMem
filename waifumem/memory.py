import time


class Memory:
    def __init__(self):
        self.timestamp = time.time()


class MessageMemory(Memory):
    """
    some message stuff idk
    """
    def __init__(self):
        super().__init__()


class ChatMemory(Memory):
    """
    Should contain a list of message memories and be a summary of
    the messages contained (?)
    """
    def __init__(self):
        super().__init__()


class UserMemory(Memory):
    """
    List of chat memories? idk if this organization is even helpful or not
    """
    def __init__(self):
        super().__init__()
