from pymilvus import MilvusClient
from fastembed import TextEmbedding
import os

class OIWikiDB :
    def __init__(
            self,
            docs_dir : str = './OI-wiki/docs/',
            db_path : str = './db/oi-wiki.db',
            embedding_model : str = 'BAAI/bge-small-zh-v1.5'
        ) :
        """
        导入一个 OI-Wiki 数据库

        @param rebuild: 是否强制重新创建

        @param docs_dir: OI-wiki/docs 位置

        @param db_path: 数据库存储位置

        @param embedding 嵌入模型名称
        """
        
        self._client = MilvusClient(db_path)
        self._embedding_model = TextEmbedding(embedding_model)
        self._collection_name = "oiwiki"
        self._docs_dir = docs_dir

        exists = self._client.has_collection(self._collection_name)
        if not exists :
            raise Exception(f"{db_path} don't have a {self._collection_name} collection! Make sure you downloaded correct db file.")
        
    def search(self, query : str) :
        qvectors = list(self._embedding_model.embed([query]))
        results = self._client.search(
            collection_name=self._collection_name, 
            data=qvectors, 
            limit=1,
            output_fields=["path"]
        )

        path = os.path.join(self._docs_dir, results[0][0].entity.path)
        with open(path, 'r') as f:
            res = f.read()

        return res
        

