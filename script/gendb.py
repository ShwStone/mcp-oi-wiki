from pymilvus import MilvusClient
from fastembed import TextEmbedding
import sys
import os
import json

def main(
        docs_dir : str = './OI-wiki/docs',
        db_path : str = './db/oi-wiki.db',
        embedding_model : str = 'BAAI/bge-small-zh-v1.5'
    ):
    client = MilvusClient(db_path)
    embedding = TextEmbedding(embedding_model)
    collection_name = "oiwiki"

    if client.has_collection(collection_name) :
        client.drop_collection(collection_name)

    contents, paths = [], []
    with open('result.jsonl', 'r') as f:
        for line in f.readlines() :
            line = line.strip('\n')
            data = json.loads(line)
            path = data["custom_id"]
            paths.append(path)
            contents.append(data["response"]["body"]["choices"][0]["message"]["content"])

            with open(os.path.join(docs_dir, path), 'r') as raw_f :
                raw = raw_f.read(512)

            paths.append(path)
            contents.append(raw)

    vectors = list(embedding.embed(contents))
    dimension = len(vectors[0])

    client.create_collection(
        collection_name=collection_name,
        dimension=dimension, 
    )

    data = [
        {"id": i, "vector": vectors[i], "path": paths[i]}
        for i in range(len(vectors))
    ]

    client.insert(
        collection_name=collection_name,
        data=data
    )

if __name__ == "__main__" :
    main(*sys.argv[1:])