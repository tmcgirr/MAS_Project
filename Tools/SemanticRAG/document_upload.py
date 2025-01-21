import os
import logging
from datetime import datetime
from uuid import uuid4
from typing import List, Dict
from pathlib import Path
from ollama import AsyncClient
from pymongo import MongoClient
import asyncio
import mimetypes

from unstructured.partition.auto import partition
from unstructured.cleaners.core import replace_unicode_quotes
from unstructured.documents.elements import Text
from unstructured.chunking.title import chunk_by_title

class DocumentUploader:
    def __init__(self, connection_string: str, model_name: str):
        """Initialize with MongoDB connection string"""
        self.client = MongoClient(connection_string)
        self.client_ollama = AsyncClient()
        self.model_name = model_name
        self.max_characters = 2000
        self.new_after_n_chars = 1500
        
        # Supported file extensions
        self.supported_extensions = {
            '.txt', '.pdf', '.doc', '.docx', '.rst', 
            '.md', '.html', '.htm', '.xml', '.json',
            '.rtf', '.epub', '.msg', '.eml'
        }

    def is_supported_file(self, filepath: str) -> bool:
        """Check if file type is supported"""
        return Path(filepath).suffix.lower() in self.supported_extensions

    async def get_embedding(self, text: str) -> List[float]:
        """Generate embeddings using Ollama's {self.model_name} model"""
        response = await self.client_ollama.embeddings(model=self.model_name, prompt=text)
        return response.embedding

    def partition_document(self, filename: str, content_type=None, strategy="hi_res"):
        """Partition document into elements"""
        return partition(filename=filename, content_type=content_type, strategy=strategy)
    
    def chunk_by_title(self, elements, max_characters=1200, new_after_n_chars=500):
        """Chunk elements by title"""
        return chunk_by_title(elements, max_characters=max_characters, new_after_n_chars=new_after_n_chars)

    def clean_elements(self, elements):
        """Clean text elements"""
        cleaned_elements = []
        for element in elements:
            text_element = Text(element.text, metadata=element.metadata)
            text_element.apply(replace_unicode_quotes)
            cleaned_elements.append(text_element)
        return cleaned_elements
    
    async def embed_elements(self, elements):
        """Embed elements using Ollama"""
        embeddings = []
        for element in elements:
            embedding = await self.get_embedding(element.text)
            element.embeddings = embedding
            embeddings.append(element)
        return embeddings

    async def upload_to_mongodb(self, elements, doc_id: str, collection_name: str, db_name: str):
        """Upload processed elements to MongoDB"""
        collection = self.client[db_name][collection_name]
        
        for i, element in enumerate(elements):
            # Create metadata dictionary
            metadata = {
                'filename': element.metadata.filename,
                'file_directory': element.metadata.file_directory,
                'page_number': element.metadata.page_number,
                'file_type': getattr(element, 'file_type', 'txt'),
                'chunk_number': i + 1,
                'total_chunks': len(elements),
                'last_modified': str(int(datetime.utcnow().timestamp() * 1000)),
                'original_doc_id': doc_id,
                'detection_origin': 'ollama_processor'
            }
            
            chunk_data = {
                "chunk_id": str(uuid4()),
                "doc_id": doc_id,
                "doc_title": metadata.get('filename', 'Document'),
                "text": element.text,
                "metadata": metadata,
                "time": metadata.get('last_modified'),
                "embedding": element.embeddings
            }

            try:
                result = collection.insert_one(chunk_data)
                print(f"Chunk {i} uploaded successfully with ID: {result.inserted_id}")
            except Exception as e:
                logging.error(f"Failed to insert chunk {i}: {str(e)}")

    async def process_document(self, 
                             filename: str, 
                             collection_name: str,
                             db_name: str,
                             content_type=None, 
                             strategy="hi_res", 
                             max_characters=1200, 
                             new_after_n_chars=500) -> Dict:
        """Process and upload a single document"""
        try:
            if not self.is_supported_file(filename):
                return {
                    "status": "error",
                    "message": f"Unsupported file type: {Path(filename).suffix}",
                    "filename": filename
                }

            doc_id = str(uuid4())
            
            # Process document
            elements = self.partition_document(filename, content_type=content_type, strategy=strategy)
            chunked_elements = self.chunk_by_title(elements, max_characters=max_characters, new_after_n_chars=new_after_n_chars)
            cleaned_elements = self.clean_elements(chunked_elements)
            embedded_elements = await self.embed_elements(cleaned_elements)
            
            # Upload to MongoDB
            await self.upload_to_mongodb(embedded_elements, doc_id, collection_name, db_name)
            
            return {
                "status": "success",
                "doc_id": doc_id,
                "filename": filename,
                "chunks_processed": len(embedded_elements)
            }
            
        except Exception as e:
            logging.error(f"Error processing document {filename}: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "filename": filename
            }

    async def process_folder(self, 
                           folder_path: str, 
                           collection_name: str,
                           db_name: str,
                           recursive: bool = True) -> Dict:
        """
        Process all supported documents in a folder
        Args:
            folder_path: Path to the folder
            collection_name: MongoDB collection name
            db_name: MongoDB database name
            recursive: Whether to process subfolders
        """
        results = {
            "successful": [],
            "failed": [],
            "skipped": [],
            "total_chunks": 0
        }

        # Get all files in the folder
        path = Path(folder_path)
        if recursive:
            files = list(path.rglob('*'))  # Recursive search
        else:
            files = list(path.glob('*'))   # Non-recursive search

        # Process each file
        for file_path in files:
            if file_path.is_file():
                if self.is_supported_file(str(file_path)):
                    result = await self.process_document(
                        filename=str(file_path),
                        collection_name=collection_name,
                        db_name=db_name
                    )
                    
                    if result["status"] == "success":
                        results["successful"].append(result)
                        results["total_chunks"] += result["chunks_processed"]
                    else:
                        results["failed"].append(result)
                else:
                    results["skipped"].append(str(file_path))

        return results
    
    async def process_input(self, 
                          input_path: str, 
                          collection_name: str,
                          db_name: str,
                          recursive: bool = True) -> Dict:
        """
        Smart entry point that handles both single files and directories
        Args:
            input_path: Path to file or directory
            collection_name: MongoDB collection name
            db_name: MongoDB database name
            recursive: Whether to process subfolders (if input is directory)
        """
        path = Path(input_path)
        
        # Handle single file
        if path.is_file():
            result = await self.process_document(
                filename=str(path),
                collection_name=collection_name,
                db_name=db_name
            )
            return {
                "type": "single_file",
                "successful": [result] if result["status"] == "success" else [],
                "failed": [result] if result["status"] == "error" else [],
                "skipped": [str(path)] if not self.is_supported_file(str(path)) else [],
                "total_chunks": result.get("chunks_processed", 0) if result["status"] == "success" else 0
            }
            
        # Handle directory
        elif path.is_dir():
            return await self.process_folder(
                folder_path=str(path),
                collection_name=collection_name,
                db_name=db_name,
                recursive=recursive
            )
            
        else:
            return {
                "type": "error",
                "message": f"Path does not exist: {input_path}",
                "successful": [],
                "failed": [],
                "skipped": [],
                "total_chunks": 0
            }