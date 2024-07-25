from typing import Literal


def message_to_text(message: dict[Literal["message", "user", "timestamp"], str | float]):
    return f"{message['user']}: {message['message']}" # convert timestamp to human-readable format and include here later (?)
