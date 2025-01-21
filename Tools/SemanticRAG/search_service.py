from typing import List, Optional
from pymongo import MongoClient
from bson import ObjectId
from langchain_community.retrievers import BM25Retriever
from ollama import AsyncClient
import logging
import asyncio
import warnings

class HybridSearch:
    def __init__(self, connection_string: str, model_name: str):
        self.client = MongoClient(connection_string)
        self.client_ollama = AsyncClient()
        self.model_name = model_name
    async def get_embedding(self, text: str) -> List[float]:
        """Generate embeddings using Ollama's nomic-embed-text model"""
        response = await self.client_ollama.embeddings(model=self.model_name, prompt=text)
        return response.embedding
        
    def _serialize_doc(self, doc: dict) -> dict:
        """Helper method to serialize MongoDB document"""
        serialized = {}
        for key, value in doc.items():
            if isinstance(value, ObjectId):
                serialized[key] = str(value)
            else:
                serialized[key] = value
        return serialized

    async def hybrid_search(self, 
                    query_text: str,
                    collection_name: str,
                    db_name: str,
                    k: int = 5) -> dict:
        """
        Performs hybrid search combining vector similarity, full-text, and BM25 ranking
        """
        try:
            # Get embedding for query text
            query_vector = await self.get_embedding(query_text)
            
            # Initialize collection
            collection = self.client[db_name][collection_name]
            
            # Vector search pipeline
            vector_pipeline = [
                {
                    "$vectorSearch": {
                        "index": "chunk_vector",
                        "path": "embedding",
                        "queryVector": query_vector,
                        "numCandidates": 100,
                        "limit": 20
                    }
                },
                {
                    "$project": {
                        "_id": 1,
                        "text": 1,
                        "doc_title": 1,
                        "metadata": 1,
                        "vector_score": {
                            "$meta": "vectorSearchScore"
                        }
                    }
                }
            ]

            # Full-text search pipeline
            text_pipeline = [
                {
                    "$search": {
                        "index": "chunk_full_text",
                        "text": {
                            "query": query_text,
                            "path": ["text", "doc_title"]
                        }
                    }
                },
                {
                    "$project": {
                        "_id": 1,
                        "text": 1,
                        "doc_title": 1,
                        "metadata": 1,
                        "text_score": {
                            "$meta": "searchScore"
                        }
                    }
                },
                {"$limit": 20}
            ]

            # Execute both searches
            vector_results = list(collection.aggregate(vector_pipeline))
            text_results = list(collection.aggregate(text_pipeline))

            # Combine results
            all_docs = {}
            
            # Process vector search results with weights
            vector_weight = 0.4
            for doc in vector_results:
                doc_id = str(doc["_id"])
                if doc_id not in all_docs:
                    all_docs[doc_id] = {
                        "_id": str(doc["_id"]),
                        "text": doc["text"],
                        "doc_title": doc.get("doc_title", ""),
                        "metadata": self._serialize_doc(doc.get("metadata", {})),
                        "vector_score": doc.get("vector_score", 0) * vector_weight,
                        "text_score": 0
                    }

            # Process text search results with weights
            text_weight = 0.6
            for doc in text_results:
                doc_id = str(doc["_id"])
                if doc_id in all_docs:
                    all_docs[doc_id]["text_score"] = doc.get("text_score", 0) * text_weight
                else:
                    all_docs[doc_id] = {
                        "_id": str(doc["_id"]),
                        "text": doc["text"],
                        "doc_title": doc.get("doc_title", ""),
                        "metadata": self._serialize_doc(doc.get("metadata", {})),
                        "vector_score": 0,
                        "text_score": doc.get("text_score", 0) * text_weight
                    }

            # Convert to list and add BM25 scoring
            combined_results = list(all_docs.values())
            texts = [doc["text"] for doc in combined_results]

            # Simplified warning handling
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if texts:
                    bm25_retriever = BM25Retriever.from_texts(texts, k=len(texts))
                    bm25_results = bm25_retriever.get_relevant_documents(query_text)

            # Create BM25 score mapping
            bm25_scores = {doc.page_content: 1/(idx + 1) for idx, doc in enumerate(bm25_results)}
            
            # Add BM25 scores and calculate final scores
            for doc in combined_results:
                doc["bm25_score"] = bm25_scores.get(doc["text"], 0)
                doc["final_score"] = (
                    doc["vector_score"] + 
                    doc["text_score"] + 
                    doc["bm25_score"]
                )

            # Sort by final score and get top k results
            combined_results.sort(key=lambda x: x.get("final_score", 0), reverse=True)
            top_results = combined_results[:k]

            return {
                "status": "success",
                "documents": [self._serialize_doc(doc) for doc in top_results]
            }

        except Exception as e:
            logging.error(f"Error in hybrid search: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }