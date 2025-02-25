# from qdrant_client import QdrantClient
# from qdrant_client.http.models import Distance, VectorParams
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.schema import BaseRetriever, Document
# from pydantic import BaseModel
# import os
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()

# class VectorStore:
#     def __init__(self, llm_choice="Groq API"):
#         qdrant_url = os.getenv("QDRANT_URL")
#         qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
#         self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

#         if llm_choice == "Groq API":
#             self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#         elif llm_choice == "HuggingFace API":
#             self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#         else:
#             raise ValueError(f"Unsupported LLM choice: {llm_choice}")

#         self.collection_name = "learning_materials"
#         # Check if the collection exists before creating it
#         if not self.client.get_collection(self.collection_name):
#             self.client.recreate_collection(
#                 collection_name=self.collection_name,
#                 vectors_config=VectorParams(size=384, distance=Distance.COSINE)
#             )

#     def add_documents(self, documents):
#         # Extract text content from Document objects
#         texts = [doc.page_content for doc in documents]
#         vectors = self.embeddings.embed_documents(texts)
#         payload = [{"text": doc.page_content} for doc in documents]
#         self.client.upsert(
#             collection_name=self.collection_name,
#             points=[{"id": i, "vector": vector, "payload": payload[i]} for i, vector in enumerate(vectors)]
#         )

#     def as_retriever(self) -> BaseRetriever:
#         class QdrantRetriever(BaseRetriever, BaseModel):
#             client: QdrantClient
#             collection_name: str
#             embeddings: HuggingFaceEmbeddings

#             def get_relevant_documents(self, query):
#                 query_vector = self.embeddings.embed_query(query)
#                 search_result = self.client.search(
#                     collection_name=self.collection_name,
#                     query_vector=query_vector,
#                     limit=5
#                 )
#                 return [Document(page_content=hit.payload["text"], metadata=hit.payload.get("metadata", {})) for hit in search_result]

#         return QdrantRetriever(client=self.client, collection_name=self.collection_name, embeddings=self.embeddings)










from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import BaseRetriever, Document
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class VectorStore:
    def __init__(self, llm_choice="Groq API"):
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        # Initialize Qdrant client
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=60)

        # Initialize embeddings model
        if llm_choice == "Groq API":
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        elif llm_choice == "HuggingFace API":
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        else:
            raise ValueError(f"Unsupported LLM choice: {llm_choice}")

        # Create or use an existing collection
        self.collection_name = "learning_materials"
        try:
            if not self.client.get_collection(self.collection_name):
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                )
        except Exception as e:
            raise RuntimeError(f"Error initializing Qdrant collection: {e}")
        # try:
        #     # Always recreate the collection
        #     self.client.recreate_collection(
        #         collection_name=self.collection_name,
        #         vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        #     )
        # except Exception as e:
        #     raise RuntimeError(f"Error initializing Qdrant collection: {e}")


    def add_documents(self, documents):
        """
        Add documents to the vector store with metadata.
        Each document must include metadata with `source_type` and `source`.
        """
        if not documents:
            raise ValueError("No documents provided for ingestion.")

        texts = [doc.page_content for doc in documents]
        vectors = self.embeddings.embed_documents(texts)
        payload = [
            {
                "text": doc.page_content,
                "source_type": doc.metadata.get("source_type", "unknown"),
                "source": doc.metadata.get("source", "unknown"),
            }
            for doc in documents
        ]

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    {"id": i, "vector": vector, "payload": payload[i]}
                    for i, vector in enumerate(vectors)
                ],
            )
        except Exception as e:
            raise RuntimeError(f"Error adding documents to Qdrant: {e}")

    def as_retriever(self) -> BaseRetriever:
        """
        Create a retriever that supports metadata filtering.
        Allows optional filtering by `source_type` or other metadata fields.
        """
        class QdrantRetriever(BaseRetriever, BaseModel):
            client: QdrantClient
            collection_name: str
            embeddings: HuggingFaceEmbeddings

            def get_relevant_documents(self, query, filter_by=None):
                """
                Retrieve relevant documents with optional metadata filtering.
                :param query: The query to search for.
                :param filter_by: Optional dictionary to filter results by metadata.
                :return: List of Document objects.
                """
                if not query:
                    raise ValueError("Query cannot be empty.")

                query_vector = self.embeddings.embed_query(query)
                search_params = {
                    "collection_name": self.collection_name,
                    "query_vector": query_vector,
                    "limit": 5,
                }

                if filter_by:
                    search_params["filter"] = {
                        "must": [{"key": k, "match": {"value": v}} for k, v in filter_by.items()]
                    }

                try:
                    search_result = self.client.search(**search_params)
                    return [
                        Document(
                            page_content=hit.payload["text"],
                            metadata={
                                "source": hit.payload.get("source", "unknown"),
                                "source_type": hit.payload.get("source_type", "unknown"),
                            },
                        )
                        for hit in search_result
                    ]
                except Exception as e:
                    raise RuntimeError(f"Error retrieving documents from Qdrant: {e}")

        return QdrantRetriever(
            client=self.client, collection_name=self.collection_name, embeddings=self.embeddings
        )


