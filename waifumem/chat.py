from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache_Q4, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2Sampler, ExLlamaV2DynamicJob
from waifumem import prompts


class Llama:
    def __init__(self, model_dir: str):
        self.config = ExLlamaV2Config(model_dir)
        self.model = ExLlamaV2(self.config)
        self.cache = ExLlamaV2Cache_Q4(self.model, max_seq_len=65536, lazy=True)
        self.model.load_autosplit(self.cache, progress = True)
        self.tokenizer = ExLlamaV2Tokenizer(self.config)
        self.generator = ExLlamaV2DynamicGenerator(
            model=self.model,
            cache=self.cache,
            tokenizer=self.tokenizer,
        )

    def generate(
            self,
            prompt: str,
            max_new_tokens: int,
            settings: ExLlamaV2Sampler.Settings = ExLlamaV2Sampler.Settings(
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
            ),
    ):
        return self.generator.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            gen_settings=settings,
            add_bos = True,
        )

    def generate_stream(
            self,
            prompt: str,
            max_new_tokens: int,
            settings: ExLlamaV2Sampler.Settings = ExLlamaV2Sampler.Settings(
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
            ),
    ):
        job = ExLlamaV2DynamicJob(
            input_ids=self.tokenizer.encode(prompt),
            max_new_tokens=max_new_tokens,
            gen_settings=settings,
            identifier=prompt # probably incorrect, fix later
        )
        self.generator.enqueue(job)

        while self.generator.num_remaining_jobs():
            results = self.generator.iterate()

            for result in results:
                if result["identifier"] != prompt:
                    continue

                yield result.get("text", "")


if __name__ == "__main__":
    llm = Llama("waifumem/models/llama-3.1-8b-instruct-exl2")

    prompt = prompts.llama3()

    llm.generate("hello world", 5)
