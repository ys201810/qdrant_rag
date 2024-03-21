# coding=utf-8
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.models import Filter


class NeuralSearcher:
    def __init__(self, collection_name):
        self.collection_name = collection_name
        # Initialize encoder model
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        # initialize Qdrant client
        self.qdrant_client = QdrantClient("http://localhost:6333")


    def search(self, text: str):
        # Convert text query into vector
        vector = self.model.encode(text).tolist()
    
        # フィルタの例
        city_of_interest = "Berlin"
        city_filter = Filter(**{
            "must": [{
                "key": "city",  # Store city information in a field of the same name
                "match": {  # This condition checks if payload field has the requested value
                    "value": city_of_interest
                }
            }]
        })

        # Use `vector` for search for closest vectors in the collection
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=None,  # city_filter # If you don't want any filters for now
            limit=5,  # 5 the most closest results is enough
        )
        # `search_result` contains found vector ids with similarity scores along with the stored payload
        # In this function you are interested in payload only
        payloads = [hit.payload for hit in search_result]
        return payloads


