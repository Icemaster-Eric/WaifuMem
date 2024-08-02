from typing import Literal


class Prompt:
    prompt: str
    add_bos: bool
    add_eos: bool
    encode_special_tokens: bool
    stop_conditions: list


class Llama3(Prompt):
    def __init__(self, messages: list[dict[Literal["role", "content"], str]], tokenizer):
        self.add_bos = False
        self.add_eos = False
        self.encode_special_tokens = True
        self.stop_conditions = [
            tokenizer.eos_token_id,
            tokenizer.single_id("<|eot_id|>"),
            tokenizer.single_id("<|start_header_id|>")
        ]

        self.prompt = "<|begin_of_text|>\n"

        for i, message in enumerate(messages):
            if i and message["role"] == "system":
                raise ValueError("The system role message must be the first.")

            self.prompt += (
                    f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n"
                    f"{message['content']}<|eot_id|>\n"
                )

            self.prompt += "<|start_header_id|>assistant<|end_header_id|>"


class MistralInstruct(Prompt):
    """
    <s>[INST] This is a system prompt.

    This is the first user input. [/INST] This is the first assistant response. </s>[INST] This is the second user input. [/INST] Second response [INST] Second Input [/INST]
    """
    def __init__(self, messages: list[dict[Literal["role", "content"], str]], tokenizer):
        self.add_bos = False
        self.add_eos = False
        self.encode_special_tokens = False # ?
        self.stop_conditions = [
            tokenizer.eos_token_id,
            tokenizer.single_id("</s>"), # not sure about this
        ]

        self.prompt = "[INST] " if len(messages) == 1 else "<s>[INST] "

        for i, message in enumerate(messages):
            if message["role"] == "system":
                if i:
                    raise ValueError("The system role message must be the first.")

                self.prompt += f"{message['content']}\n\n"
            
            elif message["role"] == "user":
                self.prompt += f"{message['content']} [/INST]"

            elif message["role"] == "assistant":
                self.prompt += f" {message['content']} [INST] "

            else:
                raise ValueError(f"Unsupported role '{message['role']}' for mistral instruct prompts.")


if __name__ == "__main__":
    print(MistralInstruct([
        {"role": "system", "content": "You are a helpful AI assistant"},
        {"role": "user", "content": "Question 1"},
        {"role": "assistant", "content": "Answer 1"},
        {"role": "user", "content": "Question 2"},
    ]).prompt)
