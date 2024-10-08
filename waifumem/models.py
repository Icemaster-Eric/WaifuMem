from waifumem.llm import Llama
from sentence_transformers import SentenceTransformer, CrossEncoder


"""
list of model paths:
waifumem/models/mistral-nemo-instruct-12b-exl2
waifumem/models/llama-3.1-8b-instruct-exl2
waifumem/models/gemma-2-27b-it-58bpw-exl2
waifumem/models/gemma-2-27b-it-5bpw-exl2
waifumem/models/gemma-2-9b-it-6bpw-exl2
"""


llm_model = Llama(
    "waifumem/models/gemma-2-9b-it-6bpw-exl2"
)
embedding_model = SentenceTransformer("Snowflake/snowflake-arctic-embed-l")
reranking_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
