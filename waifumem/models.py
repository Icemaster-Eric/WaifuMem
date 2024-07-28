from waifumem.llm import Llama
from sentence_transformers import SentenceTransformer, CrossEncoder


llm_model = Llama(
    "waifumem/models/llama-3.1-8b-instruct-exl2"
)
embedding_model = SentenceTransformer("Snowflake/snowflake-arctic-embed-l")
reranking_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
