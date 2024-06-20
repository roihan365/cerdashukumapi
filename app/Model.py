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
        
    def readDataframe(self,data : pd.DataFrame | str) -> pd.DataFrame:
        """method for reading dataframe
            method can loaded from local variable or from gdrive ural (make sure you use the public sharing url)
            argument :
            data : pd.DataFrame | str
            pd.DataFrame -> the type input is pd.DataFrame object model will read database from local variable
            str -> the method recognizer string input as url gdrive pattern : raise error when the url wasn't valid url pattern or the url not public
            return -> pd.DataFrame object
            """
        if type(data) == pd.DataFrame:
            # df = pd.read_csv(data)
            return data
        elif type(data) == str:
            startTime = time.time()
            print("fetching....")
            baseURL= "https://drive.google.com/uc?export=download&id="
            perpectPath = baseURL + data.split("/")[-2]
            df = pd.read_csv(perpectPath)
            print(f"Read data in {startTime - time.time():.2f} second")
            return df
        else:
            raise IOError("just can read from local or googledrive")
        
    def fit(self, data : pd.DataFrame | str,field : str, promptText :  str = "Pasal : ",returnData : bool = False) -> np.ndarray | None :
        """"
        make an base embedings from pasal kuhperdata, the embedings stored in local before added to vector database
        argument:
        data : pd.Dataframe 
        data : str
        -> the argument is derivate from  readDataframe method
        prompt : str
        -> prompting text that we can inject more domain spesific information to our base embedings. the expectation when we add the prompting
           is model can extract and transform more relevant embeddings transformation
        return -> None or np.ndarray
        """
        self.dataframe = self.readDataframe(data)
        dataToEncodeEmbedings = self.dataframe[field].apply(lambda x : x.lower()).tolist()
        self.baseEmbedings = self.model.encode(dataToEncodeEmbedings,prompt=promptText, show_progress_bar=True,normalize_embeddings=True,device=self.device)
        if returnData:
            return self.baseEmbedings

    def fit_predict(self, text : str, topk : int = 10) -> tuple[np.ndarray[int]]:
        """
        implement semantic search that we can use for searching which one relevant pasal with user text
        text : str
        -> define as input from users
        topk : int
        -> how much we want look at semantic embedings that was similar with pasal kuhperdata
        """
        vector =  self.model.encode(text)
        score = np.flip(np.sort(cosine_similarity([vector],self.baseEmbedings)[0][:topk]))
        rank =  np.flip(cosine_similarity([vector],self.baseEmbedings).argsort(-1)[0])[:topk]
        return rank, score