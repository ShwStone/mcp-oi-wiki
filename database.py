from pymilvus import MilvusClient
import numpy as np
from fastembed import TextEmbedding
import os

class OIWikiDB :
    def __init__(
            self, 
            rebuild : bool = False, 
            docs_dir : str = './OI-wiki/docs/',
            db_path : str = './db/oi-wiki.db',
            embedding_model : str = 'BAAI/bge-small-zh-v1.5',
            effective_length : int = 16
        ) :
        """
        创建/导入一个 OI-Wiki 数据库

        @param rebuild: 是否强制重新创建

        @param docs_dir: OI-wiki/docs 位置

        @param db_path: 数据库存储位置

        @param embedding 模型名称

        @effective_length 索引的最小段落长度
        """
        
        self.client = MilvusClient(db_path)
        self.embedding_model = TextEmbedding(embedding_model)
        self._effective_length = effective_length
        self._collection_name = "oiwiki"

        exists = self.client.has_collection(self._collection_name)

        if exists :
            if rebuild :
                self.client.drop_collection(self._collection_name)
                self._load_oi_wiki(docs_dir)
        else :
            self._load_oi_wiki(docs_dir)

    def _load_oi_wiki(self, docs_dir : str) :
        contents = []
        paths = []

        for subject_name in os.listdir(docs_dir):
            subject_path = os.path.join(docs_dir, subject_name)
            if not os.path.isdir(subject_path) :
                continue

            for file in os.listdir(subject_path):
                if not file.endswith('.md') :
                    continue

                file_path = os.path.join(subject_path, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    # content = [line.strip() for line in f]
                    # content = [
                    #     line 
                    #     for line in content
                    #     if len(line) - line.count(' ') >= self._effective_length
                    # ]
                    content = [f.read().strip()]
                    contents += content
                    paths += [file_path] * len(content)
        
        vectors = list(self.embedding_model.embed(contents))
        dimension = len(vectors[0])

        self.client.create_collection(
            collection_name=self._collection_name,
            dimension=dimension, 
        )

        data = [
            {"id": i, "vector": vectors[i], "path": paths[i]}
            for i in range(len(vectors))
        ]

        self.client.insert(
            collection_name=self._collection_name,
            data=data
        )
        
    def search(self, query : str | list[str]) :
        if type(query) == str :
            query = [query]

        qvectors = list(self.embedding_model.embed(query))
        res = self.client.search(
            collection_name=self._collection_name, 
            data=qvectors, 
            limit=2,
            output_fields=["path"]
        )

        return res
