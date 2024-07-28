from typing import Literal


def llama3(messages: list[dict[Literal["role", "content"], str]]) -> str:
    """Generates a llama 3 prompt given a list of messages.

    Args:
        messages (list[dict[Literal[&quot;role&quot;, &quot;content&quot;], str]]): Conversation as a list of messages

    Returns:
        str: The generated prompt.
    """
    prompt = "<|begin_of_text|>"

    for message in messages:
        prompt += (
            f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n"
            f"{message['content']}<|eot_id|>\n"
        )

    prompt += "<|start_header_id|>assistant<|end_header_id|>"

    return prompt


if __name__ == "__main__":
    print(llama3([
        {"role": "system", "content": "You are a helpful AI assistant"},
        {"role": "user", "content": "Question 1"},
        {"role": "assistant", "content": "Answer 1"},
        {"role": "user", "content": "Question 2"},
    ]))
