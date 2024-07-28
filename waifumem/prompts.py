from typing import Literal


class Prompt:
    prompt: str
    add_bos: bool
    add_eos: bool
    encode_special_tokens: bool
    stop_conditions: list


class Llama3(Prompt):
    def __init__(self, messages: list[dict[Literal["role", "content"], str]], tokenizer):
        self.prompt = "<|begin_of_text|>\n"

        for message in messages:
            self.prompt += (
                    f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n"
                    f"{message['content']}<|eot_id|>\n"
                )

            self.prompt += "<|start_header_id|>assistant<|end_header_id|>"

        self.add_bos = False
        self.add_eos = False
        self.encode_special_tokens = True
        self.stop_conditions = [
            tokenizer.eos_token_id,
            tokenizer.single_id("<|eot_id|>"),
            tokenizer.single_id("<|start_header_id|>")
        ]


if __name__ == "__main__":
    print(Llama3([
        {"role": "system", "content": "You are a helpful AI assistant"},
        {"role": "user", "content": "Question 1"},
        {"role": "assistant", "content": "Answer 1"},
        {"role": "user", "content": "Question 2"},
    ]).prompt)
