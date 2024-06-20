import torch
from sentence_transformers import SentenceTransformer
import time
from qdrant_client import QdrantClient
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class IndoSbertModel(object):
    """
    This is a sentence-transformers model: It maps sentences & paragraphs to a 256 dimensional
    dense vector space and can be used for tasks like clustering or semantic search.
    IndoSBERT is a modification of https://huggingface.co/indobenchmark/indobert-large-p1
    that has been fine-tuned using the siamese network scheme inspired by SBERT (Reimers et al., 2019).
    This model was fine-tuned with the STS Dataset (2012-2016) which was machine-translated into Indonesian languange.
    """
    def __init__(self, model : SentenceTransformer = SentenceTransformer("denaya/indoSBERT-large")) :
        now = time.time()
        self.model = model 
        self.device = "cuda" if torch.cuda.is_available() else "cpu" #checking if the device that used for running model contains GPU
        print(f"loaded {now - time.time() :.2f} second")

    def getDatabaseVector(self,sentence : str, promptText :  str = "Pasal : ", returnPredict : bool = False):
        """
        implementas the method for fetching the embedings to databse vector and cosine predict value from database vectors
        """
        qdrant_host="https://651082da-8166-4011-ab66-0ed93f9d3f5a.us-east4-0.gcp.cloud.qdrant.io:6333"
        api_key="kxureD9-xh0VsZOxs7GuSQxpGsU9ffO_gWUB-HRyVDcC100zpjiXEQ"

        qdrant_client = QdrantClient(url=qdrant_host, api_key=api_key, timeout=55)

        input_embedding = self.model.encode(sentence,prompt=promptText, show_progress_bar=True,normalize_embeddings=True,device=self.device)
        if(returnPredict):
            search_result = qdrant_client.search(
                collection_name="indosbert_lower",
                query_vector=input_embedding,
                limit=5,
            )
            return search_result