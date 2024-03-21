# coding=utf-8
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance


def make_and_save_embeddings(model: SentenceTransformer, df: pd.DataFrame):
    vectors = model.encode(
        [row.alt + ". " + row.description for row in df.itertuples()],
        show_progress_bar=True,
    )

    print(vectors.shape)
    np.save("../data/startups/startup_vectors.npy", vectors, allow_pickle=False)

def main():
    sentence_transformer_model = "all-MiniLM-L6-v2"
    model = SentenceTransformer(
        sentence_transformer_model,
        device="cpu"  # "cuda"だとGPU
    )

    df = pd.read_json("../data/startups/startups_demo.json", lines=True)

    # エンべディングの作成と保存
    # make_and_save_embeddings(model, df)

    # クライアントの作成(shell/start_qdrant.shを実行後)
    qdrant_client = QdrantClient("http://localhost:6333")

    qdrant_client.recreate_collection(
        collection_name="startups",
        vectors_config=VectorParams(
            size=model.get_sentence_embedding_dimension(),
            distance=Distance.COSINE),
    )

    fd = open("../data/startups/startups_demo.json")
    payload = map(json.loads, fd)
    vectors = np.load("../data/startups/startup_vectors.npy")

    qdrant_client.upload_collection(
        collection_name="startups",
        vectors=vectors,
        payload=payload,
        ids=None,  # Vector ids will be assigned automatically
        batch_size=256,  # How many vectors will be uploaded in a single request?
    )

if __name__ == '__main__':
    main()
