import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2DynamicGenerator
from exllamav2.generator.filters import ExLlamaV2PrefixFilter
from lmformatenforcer.integrations.exllamav2 import ExLlamaV2TokenEnforcerFilter
from lmformatenforcer import JsonSchemaParser
from pydantic import BaseModel, conlist
from typing import Literal
import json

model_dir = "waifumem/models/llama-3.1-8b-instruct-exl2"
config = ExLlamaV2Config(model_dir)
config.arch_compat_overrides()
model = ExLlamaV2(config)
cache = ExLlamaV2Cache(model, max_seq_len = 32768, lazy = True)
model.load_autosplit(cache, progress = True)

print("Loading tokenizer...")
tokenizer = ExLlamaV2Tokenizer(config)

# Initialize the generator with all default parameters

generator = ExLlamaV2DynamicGenerator(
    model = model,
    cache = cache,
    tokenizer = tokenizer,
)

# JSON schema

class RememberMessage(BaseModel):
    remember: bool

schema_parser = JsonSchemaParser(RememberMessage.schema())

# Create prompts with and without filters

def classify_prompt(message: str):
    return """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an AI assistant that classifies whether a message should be remembered or not, and returns the output in JSON<|eot_id|><|start_header_id|>user<|end_header_id|>

Raiden is an infinitely more fun boss than signora<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{"remember": true}<|eot_id|><|start_header_id|>user<|end_header_id|>

Hi Lucy goosey :goose:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{"remember": false}<|eot_id|><|start_header_id|>user<|end_header_id|>

Woah who were they?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{"remember": false}<|eot_id|><|start_header_id|>user<|end_header_id|>

Lmfao that's dark<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{"remember": false}<|eot_id|><|start_header_id|>user<|end_header_id|>

artifact farming is hell on earth :sob:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{"remember": true}<|eot_id|><|start_header_id|>user<|end_header_id|>

That's a honkai star rail term<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{"remember": true}<|eot_id|><|start_header_id|>user<|end_header_id|>

""" + message + """<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

i_prompts = [
    classify_prompt("BROTHER THATS KACHINA :skull::skull: And Im matching with my best friends"),
    classify_prompt("wtf you got a Nahida pfp, weren't you her biggest hater. what has happened to you"),
    classify_prompt("nothing much, just relaxing n stuff"),
    classify_prompt("Never tried it let me see"),
    classify_prompt("Do I send my uid here"),
]

prompts = []
filters = []

for p in i_prompts:
    prompts.append(p)
    filters.append(None)
    prompts.append(p)
    filters.append([
        ExLlamaV2TokenEnforcerFilter(schema_parser, tokenizer),
        ExLlamaV2PrefixFilter(model, tokenizer, ["{", " {"])
    ])

# Generate

print("Generating...")

outputs = generator.generate(
    prompt = prompts,
    filters = filters,
    filter_prefer_eos = True,
    max_new_tokens = 10,
    add_bos = True,
    stop_conditions = [tokenizer.eos_token_id],
    completion_only = True
)

# Print outputs:

for i in range(len(i_prompts)):
    print("---------------------------------------------------------------------------------")
    print(i_prompts[i].strip())
    print()
    print("Without filter:")
    print("---------------")
    print(outputs[i * 2])
    print()
    print("With filter:")
    print("------------")
    print(json.dumps(json.loads(outputs[i * 2 + 1]), indent = 4).strip())
    print()
