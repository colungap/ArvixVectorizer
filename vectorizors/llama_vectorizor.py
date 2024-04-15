from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings



class LlamaEmbeder:
    """
    a handy wrapper for the llama model currently very spartan but it could be a useful way to store pipeline specific
    functions/methods to allow for simple reuse.
    """

    def __init__(self):
        self.model = Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def batch_embedding(self, doc: [str]):
        embedding = self.model.get_text_embedding_batch(doc)
        return embedding

    def embedding(self, doc: str):
        embedding = self.model.get_text_embedding(doc)
        return embedding
