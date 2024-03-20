# coding=utf-8
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
from semantic_tutorial_data import get_tutorial_data


def main():
    # 1. data preparation
    documents = get_tutorial_data()

    # 2. get embedding model
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # 3. define storage location and create a collection
    qdrant = QdrantClient(":memory:")
    # qdrant = QdrantClient("localhost", port=6333)

    qdrant.recreate_collection(  # recreate_collectionはcollectionが既に存在すれば削除して作成し直す。
        collection_name="my_books",
        vectors_config=models.VectorParams(
            size=encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
            distance=models.Distance.COSINE,
        ),
    )

    # 4. upload data to collection
    qdrant.upload_points(
        collection_name="my_books",
        points=[
            models.PointStruct(
                id=idx, vector=encoder.encode(doc["description"]).tolist(), payload=doc
            )
            for idx, doc in enumerate(documents)
        ],
    )

    # 5. ask the engine a question and
    hits = qdrant.search(
        collection_name="my_books",
        query_vector=encoder.encode("alien invasion").tolist(),
        limit=3,
    )
    # print the results
    for hit in hits:
        print(hit.payload, "score:", hit.score)

    # narrow down the query
    hits = qdrant.search(
        collection_name="my_books",
        query_vector=encoder.encode("alien invasion").tolist(),
        query_filter=models.Filter(
            must=[models.FieldCondition(key="year", range=models.Range(gte=2000))]
        ),
        limit=1,
    )
    for hit in hits:
        print(hit.payload, "score:", hit.score)


if __name__ == '__main__':
    main()
