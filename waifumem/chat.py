import torch

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache_Q4,
    ExLlamaV2Tokenizer,
)

from exllamav2.generator import (
    ExLlamaV2StreamingGenerator,
    ExLlamaV2Sampler
)


class Llama:
    def __init__(self, model_dir: str, prompt_format):
        config = ExLlamaV2Config(model_dir)
        model = ExLlamaV2(config)
        tokenizer = ExLlamaV2Tokenizer(config)

        cache = ExLlamaV2Cache_Q4(model, lazy = not model.loaded)

        generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)

        settings = ExLlamaV2Sampler.Settings(
            temperature = 0.95, # Sampler temperature (1 to disable)
            top_k = 50, # Sampler top-K (0 to disable)
            top_p = 0.8, # Sampler top-P (0 to disable)
            top_a = 0.0, # Sampler top-A (0 to disable)
            typical = 0.0, # Sampler typical threshold (0 to disable)
            skew = 0.0, # Skew sampling (0 to disable)
            token_repetition_penalty = 1.01, # Sampler repetition penalty (1 to disable)
            token_frequency_penalty = 0.0, # Sampler frequency penalty (0 to disable)
            token_presence_penalty = 0.0, # Sampler presence penalty (0 to disable)
            smoothing_factor = 0.0, # Smoothing Factor (0 to disable)
        )

    def chat_completion(self, ) -> str:
        pass

    def get_tokenized_context(self, max_len):
        global user_prompts, responses_ids

        while True:
            context = torch.empty((1, 0), dtype=torch.long)

            for turn in range(len(user_prompts)):

                up_text = format_prompt(user_prompts[turn], context.shape[-1] == 0)
                up_ids = encode_prompt(up_text)
                context = torch.cat([context, up_ids], dim=-1)

                if turn < len(responses_ids):
                    context = torch.cat([context, responses_ids[turn]], dim=-1)

            if context.shape[-1] < max_len: return context

            # If the context is too long, remove the first Q/A pair and try again. The system prompt will be moved to
            # the first entry in the truncated context

            user_prompts = user_prompts[1:]
            responses_ids = responses_ids[1:]


class PromptLlama3:
    botname = "Chatbot"
    username = "User"
    description = "Llama3-instruct models"

    def __init__(self):
        pass

    def first_prompt(self):
        return \
            """<|start_header_id|>system<|end_header_id|>\n\n""" + \
            """<|system_prompt|><|eot_id|>""" + \
            """<|start_header_id|>user<|end_header_id|>\n\n""" + \
            """<|user_prompt|><|eot_id|>""" + \
            """<|start_header_id|>assistant<|end_header_id|>"""

    def subs_prompt(self):
        return \
            """<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n""" + \
            """<|user_prompt|><|eot_id|>""" + \
            """<|start_header_id|>assistant<|end_header_id|>"""

    def stop_conditions(self, tokenizer: ExLlamaV2Tokenizer):
        return [
            tokenizer.eos_token_id,
            tokenizer.single_id("<|eot_id|>"),
            tokenizer.single_id("<|start_header_id|>")
        ]

    def encoding_options(self):
        return False, False, True


username = "Eric"
botname = "Raine"
system_prompt = "You are Raine, a AI vtuber with a kuudere personality. You are a shy girl who doesn't like to talk very much. However, you still make sarcastic remarks and tease others sometimes. Never talk in third person. Never describe your actions. Always respond in first person as Raine. You are talking to Eric."

prompt_format = PromptLlama3()
prompt_format.botname = botname
prompt_format.username = username

# Initialize model and tokenizer

config = ExLlamaV2Config("waifumem/models/llama-3.1-8b-instruct-exl2")
model = ExLlamaV2(config)
tokenizer = ExLlamaV2Tokenizer(config)

cache = ExLlamaV2Cache_Q4(model, lazy = not model.loaded)


# Chat context

def format_prompt(user_prompt, first):
    global system_prompt, prompt_format

    if first:
        return prompt_format.first_prompt() \
            .replace("<|system_prompt|>", system_prompt) \
            .replace("<|user_prompt|>", user_prompt)
    else:
        return prompt_format.subs_prompt() \
            .replace("<|user_prompt|>", user_prompt)

def encode_prompt(text):
    global tokenizer, prompt_format

    add_bos, add_eos, encode_special_tokens = prompt_format.encoding_options()
    return tokenizer.encode(text, add_bos = add_bos, add_eos = add_eos, encode_special_tokens = encode_special_tokens)

user_prompts = []
responses_ids = []

def get_tokenized_context(max_len):
    global user_prompts, responses_ids

    while True:

        context = torch.empty((1, 0), dtype=torch.long)

        for turn in range(len(user_prompts)):

            up_text = format_prompt(user_prompts[turn], context.shape[-1] == 0)
            up_ids = encode_prompt(up_text)
            context = torch.cat([context, up_ids], dim=-1)

            if turn < len(responses_ids):
                context = torch.cat([context, responses_ids[turn]], dim=-1)

        if context.shape[-1] < max_len: return context

        # If the context is too long, remove the first Q/A pair and try again. The system prompt will be moved to
        # the first entry in the truncated context

        user_prompts = user_prompts[1:]
        responses_ids = responses_ids[1:]


# Generator

generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)

settings = ExLlamaV2Sampler.Settings(
    temperature = 0.95, # Sampler temperature (1 to disable)
    top_k = 50, # Sampler top-K (0 to disable)
    top_p = 0.8, # Sampler top-P (0 to disable)
    top_a = 0.0, # Sampler top-A (0 to disable)
    typical = 0.0, # Sampler typical threshold (0 to disable)
    skew = 0.0, # Skew sampling (0 to disable)
    token_repetition_penalty = 1.01, # Sampler repetition penalty (1 to disable)
    token_frequency_penalty = 0.0, # Sampler frequency penalty (0 to disable)
    token_presence_penalty = 0.0, # Sampler presence penalty (0 to disable)
    smoothing_factor = 0.0, # Smoothing Factor (0 to disable)
)

max_response_tokens = 500 # max tokens / response

# Stop conditions

sc = prompt_format.stop_conditions(tokenizer)
sc = [x for x in sc if x]
generator.set_stop_conditions(sc)

# Main loop

while True:
    up = input(username + ": ").strip()
    print()

    # Add to context

    user_prompts.append(up)

    # Send tokenized context to generator

    active_context = get_tokenized_context(model.config.max_seq_len)
    generator.begin_stream_ex(active_context, settings)

    # Stream response

    response_tokens = 0
    response_text = ""
    responses_ids.append(torch.empty((1, 0), dtype = torch.long))

    while True:

        # Get response stream

        res = generator.stream_ex()
        chunk = res["chunk"]
        eos = res["eos"]
        tokens = res["chunk_token_ids"]

        if len(response_text) == 0: chunk = chunk.lstrip()
        response_text += chunk
        responses_ids[-1] = torch.cat([responses_ids[-1], tokens], dim = -1)

        print(chunk, end = "")

        # If response is too long, cut it short, and append EOS if that was a stop condition

        response_tokens += 1
        if response_tokens == max_response_tokens:

            if tokenizer.eos_token_id in generator.stop_tokens:
                responses_ids[-1] = torch.cat([responses_ids[-1], tokenizer.single_token(tokenizer.eos_token_id)], dim = -1)

            print()
            print(col_error + f" !! Response exceeded {max_response_tokens} tokens and was cut short." + col_default)
            break

        # EOS signal returned

        if eos:
            break
