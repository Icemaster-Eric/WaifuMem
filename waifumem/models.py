from llama_cpp import Llama
from sentence_transformers import SentenceTransformer


llm_model = Llama(
    model_path="waifumem/models/gemma-2-27b-it-Q5_K_L.gguf",
    chat_format="gemma",
    n_ctx=8192,
    n_gpu_layers=-1,
    use_mmap=False,
    verbose=False
)
embedding_model = SentenceTransformer("Snowflake/snowflake-arctic-embed-l")
