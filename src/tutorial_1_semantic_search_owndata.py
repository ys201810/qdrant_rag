# coding=utf-8
import time
import pathlib
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer


def get_own_data(target_path: pathlib.Path):
    book_datas = []
    for target_file in target_path.glob("*.md"):
        text = ''
        book_data = {}
        with open(target_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.rstrip()
                if i == 0:  # 先頭に発売年があるのでこれを取得。
                    year = line.split(':')[1]
                else:
                    text += line
            book_data['title'] = target_file.stem
            book_data['year'] = year
            book_data['description'] = text
        book_datas.append(book_data)
    return book_datas


def main():
    # 0. setting
    base_path = pathlib.Path(__file__).resolve().parents[1]
    collection_name = 'own_japanese_books'
    sentence_transformer_model = "intfloat/multilingual-e5-small"
    # 1. data preparation
    documents = get_own_data(base_path / 'data' / 'zenn_data')

    # 2. get embedding model
    encoder = SentenceTransformer(sentence_transformer_model)  # ("all-MiniLM-L6-v2")

    # 3. define storage location and create a collection
    # qdrant = QdrantClient(":memory:")
    qdrant = QdrantClient("localhost", port=6333)

    qdrant.recreate_collection(  # recreate_collectionはcollectionが既に存在すれば削除して作成し直す。
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
            distance=models.Distance.COSINE,
        ),
    )

    # 4. upload data to collection
    start_time = time.time()
    qdrant.upload_points(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=idx, vector=encoder.encode(doc["description"]).tolist(), payload=doc
            )
            for idx, doc in enumerate(documents)
        ],
    )
    end_time = time.time()
    print(f"Data upload took {round(end_time - start_time, 5)} seconds")

    # 5. ask the engine a question and
    start_time = time.time()

    query = '富を得るための方法が知りたいです。'
    hits = qdrant.search(
        collection_name=collection_name,
        query_vector=encoder.encode(query).tolist(),
        limit=3,
    )
    end_time = time.time()
    print(f"Data search took {round(end_time - start_time, 5)} seconds")

    # print the results
    for hit in hits:
        print(hit.payload, "score:", hit.score)

    # narrow down the query
    hits = qdrant.search(
        collection_name=collection_name,
        query_vector=encoder.encode("alien invasion").tolist(),
        query_filter=models.Filter(
            must=[models.FieldCondition(key="year", range=models.Range(gte=2018))]
        ),
        limit=1,
    )
    for hit in hits:
        print(hit.payload, "score:", hit.score)


if __name__ == '__main__':
    main()