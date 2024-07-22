import time


class Memory:
    def __init__(self):
        self.timestamp = time.time()


class MessageMemory(Memory):
    """
    some desc here
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
    some desc here
    """
    def __init__(self):
        super().__init__()
