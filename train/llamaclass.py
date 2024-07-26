from typing import Optional, List
from llama_cpp import Llama, LogitsProcessorList
from lmformatenforcer import CharacterLevelParser, JsonSchemaParser
from lmformatenforcer.integrations.llamacpp import build_llamacpp_logits_processor, build_token_enforcer_tokenizer_data
from pydantic import BaseModel


"""llm = Llama(
    model_path="train/Meta-Llama-3-70B-Instruct-IQ2_XS.gguf",
    chat_format="llama-3",
    n_ctx=8192,
    n_gpu_layers=-1,
)"""


DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant for classifying messages and outputting the results in JSON."""


def get_prompt(message: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


"""tokenizer_data = build_token_enforcer_tokenizer_data(llm)

def llamacpp_with_character_level_parser(prompt: str, character_level_parser: Optional[CharacterLevelParser]) -> str:
    logits_processors: Optional[LogitsProcessorList] = None
    if character_level_parser:
        logits_processors = LogitsProcessorList([build_llamacpp_logits_processor(tokenizer_data, character_level_parser)])
    
    output = llm(prompt, logits_processor=logits_processors, max_tokens=100)
    text: str = output['choices'][0]['text']
    return text"""


class AnswerFormat(BaseModel):
    relevant: bool


question = """Please classify the following. You MUST answer using the following json schema: """
question_with_schema = f"{question}{AnswerFormat.schema_json()}"
prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant for classifying messages and outputting the results in JSON. For each of the following messages, classify their relevance and give outputs<|eot_id|><|start_header_id|>user<|end_header_id|>

question here<|eot_id|><|start_header_id|>assistant<|end_header_id|>

response here<|eot_id|><|start_header_id|>user<|end_header_id|>

question here<|eot_id|><|start_header_id|>assistant<|end_header_id|>

response here<|eot_id|><|start_header_id|>user<|end_header_id|>

question here<|eot_id|><|start_header_id|>assistant<|end_header_id|>

response here<|eot_id|><|start_header_id|>user<|end_header_id|>

question here<|eot_id|><|start_header_id|>assistant<|end_header_id|>

response here<|eot_id|><|start_header_id|>user<|end_header_id|>"""
print(prompt)

#result = llamacpp_with_character_level_parser(prompt, JsonSchemaParser(AnswerFormat.schema()))


#print(result)
