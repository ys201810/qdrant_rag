docker run -p 6333:6333 \
    -v $(pwd)/../qdrant_storage:/qdrant/storage \
    qdrant/qdrant