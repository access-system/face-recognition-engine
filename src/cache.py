import hashlib
import time

import redis
from redis.commands.search.field import TagField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query


class VerifiedEmbeddingsCache:
    def __init__(self, log, host='localhost', port=6379, db=0, password=''):
        self.log = log

        self.EMBEDDING_INDEX_NAME = "embeddings_index"
        self.EMBEDDING_DIMENSIONS = 512

        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.init_client()

    def init_client(self):
        self.client = redis.Redis(host=self.host, port=self.port, db=self.db)
        self.wait_until_ready()
        self.create_index()

    def wait_until_ready(self):
        while True:
            if self.is_ready():
                self.log.info("Cache is ready.")
                break

            self.log.info("Waiting for cache to be ready...")
            time.sleep(1)

    def is_ready(self) -> bool:
        try:
            self.client.ping()
            return True
        except redis.exceptions.ConnectionError:
            return False

    def create_index(self):
        try:
            self.client.ft(self.EMBEDDING_INDEX_NAME).info()
        except:
            schema = (
                TagField("name"),
                VectorField("embedding", "FLAT", {
                    "TYPE": "FLOAT32",
                    "DIM": self.EMBEDDING_DIMENSIONS,
                    "DISTANCE_METRIC": "COSINE"
                }),
            )
            definition = IndexDefinition(prefix=["doc:"], index_type=IndexType.HASH)
            self.client.ft(self.EMBEDDING_INDEX_NAME).create_index(schema, definition=definition)

    def store_embedding(self, embedding, name) -> None:
        pipe = self.client.pipeline()
        key = f"doc:{hashlib.sha256(embedding.tobytes()).hexdigest()}"

        tag_name = name.replace(" ", "|")
        pipe.hset(key, mapping={
            "name": tag_name,
            "embedding": embedding.tobytes()
        })
        pipe.expire(key, 3600)  # Set expiration time to 1 hour
        pipe.execute()


    def verify_embedding(self, embedding):
        base_query = f'*=>[KNN 1 @embedding $vec_param AS vector_score]'
        query = Query(base_query).sort_by("vector_score").return_fields("name", "vector_score").dialect(2)

        vec_param = embedding.astype('float32').tobytes()
        params_dict = {"vec_param": vec_param}

        results = self.client.ft(self.EMBEDDING_INDEX_NAME).search(query, query_params=params_dict)

        if results.total > 0:
            score = results.docs[0].vector_score
            if 1.0 - float(score) > 0.5:  # Threshold for verification
                data = {
                    "name": results.docs[0]["name"],
                    "accuracy": (1.0 - float(score)) * 100
                }
                return True, data
        return False, None
